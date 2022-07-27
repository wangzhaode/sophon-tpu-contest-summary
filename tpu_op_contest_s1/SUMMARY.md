# 算子打榜赛赛题总结

本次比赛的思路与流程与第一届比赛相同，整体流程为：
1. 分析需要实现的算子，查阅文档，使用tpu提供的函数实现一个不考虑片上存储限制的naive版本；
2. 通过循环分块（tiling)来解决存储限制，能够正确跑通所有case;
3. 寻找优化点进行优化，包括通用优化（如：gdma和bdc并行）和针对case的优化(如：并行度优化，io优化)；

## 1. 问题分析
本次赛题共有4个算子，分别为avgpool, reducesum, rgb2bgr和transpose。其中avgpool和reducesum是计算密集型算子，rgb2bgr和transpose是访存密集型算子。
我们首先分析实现这4种算子需要使用的能力：
1. avgpool: 需要一个avgpool函数，观察测试数据发现需要具备ceil_mode，count_include_pad功能的avgpool;
2. reducesum: 需要一个reducesum函数，观察测试数据发现需要支持各类axis;
3. rgb2bgr: 需要一个rgb2bgr函数；
4. transpose: 需要一个transpose函数，需要支持各种permute参数；

## 2. 实现思路
阅读 [okkernel文档](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/okkernel/html/index.html) ，寻找满足第1节中的函数，结果如下：
1. avgpool: 能够找到 [avgpool](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/okkernel/html/usage/refined.html#okk-bdc-avg-pool2d) 函数，具备
基础的avgpool功能；缺失ceil_mode和count_include_pad的属性控制，同时限制stride的范围为[1, 15], 对与这3个问题，思路如下：
    - ceil_mode属性用来确定输出的宽高；文档中的avgpool中该属性固定为`ceil_model=false`；对问题中的case进行分析可以发现，
      `ceil_mode=true`的case其实际输出与`ceil_mode=false`没有区别，所以该属性可以忽略，不需要特殊处理；
    - count_include_pad属性用来决定求均值时是否计算pad的部分；文档中的avgpool中该属性固定为 `count_include_pad=true` ；
      对问题中的case进行分析可以发现，当 `pads = 0` 时，该属性不起作用；当 `pads != 0` 时，该属性会对结果产生影响。
      因此需要对 `pads != 0 & count_include_pad=false` 的case进行额外处理，处理思路有2种：1. 改变角落和边缘部分的kernel_size；
      2. 对avgpool的结果使用 [okk_bdc_mul_C](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/okkernel/html/usage/refined.html#okk-bdc-mul-c)
      将结果乘以一个scale还原为不计算pad的值；
    - case 1的 `stride_h = stride_w = 16` 是avgpool不支持的参数类型，对于这种情况，可以将计算沿着h和w方向分块实现；
2. reducesum: 发现文档中并没有reduce函数，但是[avgpool](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/okkernel/html/usage/refined.html#okk-bdc-avg-pool2d)
可以实现reduce功能, 对结果执行[okk_bdc_mul_C](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/okkernel/html/usage/refined.html#okk-bdc-mul-c)获得sum值。
可以发现avgpool能够处理reduce的axis数目为1和2的情况，无法处理axis为3的情况，对于此类情况可以执行2次avgpool来实现；
3. rgb2br: 文档中没有提供rgb2brg函数和transpose函数，但是 [okk_gdma_32bit_cpy_S2S](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/okkernel/html/usage/refined.html#okk-gdma-32bit-cpy-s2s)
函数可以看到，其支持指定`shape`, `dst_stride`, `src_stride` 的内存拷贝操作。因此可以将stride的最低维设置为3来实现对图片的单通道拷贝。
4. transpose: 同3，可以通过设置`src_stride`的最低维度来实现转置操作。

## 3. 算子实现
根据第2节思路，对4中算子进行实现。忽略参数设置部分，核心实现如下：
### 3.1 avgpool
1. 首先实现最基础的naive版本：
```cpp
// load
okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr, &shape, NULL, NULL);
// compute
okk_bdc_avg_pool2d(output_addr, input_addr, &shape, param->kernel_h, param->kernel_w, &pad, &strides);
// store
okk_gdma_32bit_cpy_L2S(param->output_addr, output_addr, &avg_shape, NULL, NULL);
```
2. 然后针对片上存储限制，对batch进行分块计算的版本：
```cpp
for (int i = 0; i < n; i++) {
    okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr + i * iblock, &shape, NULL, NULL);
    okk_bdc_avg_pool2d(output_addr, input_addr, &shape, param->kernel_h, param->kernel_w, &pad, &strides);
    okk_gdma_32bit_cpy_L2S(param->output_addr + i * oblock, output_addr, &avg_shape, NULL, NULL);
}
```
3. 针对 `stride > 15` ，对h,w进行分块计算的版本：
```cpp
// left up
okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr, &shape, NULL, &istride);
okk_bdc_avg_pool2d(output_addr, input_addr, &shape, param->kernel_h, param->kernel_w, NULL, NULL);
okk_gdma_32bit_cpy_L2S(param->output_addr, output_addr, &avg_shape, &ostride, NULL);
// right up
okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr + shape.w * sizeof(float), &shape, NULL, &istride);
okk_bdc_avg_pool2d(output_addr, input_addr, &shape, param->kernel_h, param->kernel_w, NULL, NULL);
okk_gdma_32bit_cpy_L2S(param->output_addr + sizeof(float), output_addr, &avg_shape, &ostride, NULL);
// left bottom ...
// right bottom ...
```
4. 针对 `pads != 0 & count_include_pad=false` 的计算版本(第二种思路)：
```cpp
// load
okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr, &shape, NULL, NULL);
// avgpool
okk_bdc_avg_pool2d(output_addr, input_addr, &shape, param->kernel_h, param->kernel_w, &pad, &strides);
// scale corner
// left up corner
okk_bdc_mul_C(output_addr, output_addr, corner_scale, &corner_shape, &avg_stride, &avg_stride);
// right up corner
okk_bdc_mul_C(output_addr + w_offset, output_addr + w_offset, corner_scale, &corner_shape, &avg_stride, &avg_stride);
// left bottom corner ...
// right bottom corner ...
// scale edge
// top edge
okk_bdc_mul_C(output_addr + offset_w, output_addr + offset_w, edge_scale, &edge_shape_w, &avg_stride, &avg_stride);
// bottom edge
okk_bdc_mul_C(output_addr + h_offset + offset_w, output_addr + h_offset + offset_w, edge_scale, &edge_shape_w, &avg_stride, &avg_stride);
// left edge ...
// right edge ...
// store
okk_gdma_32bit_cpy_L2S(param->output_addr, output_addr, &avg_shape, NULL, NULL);
```
### 3.2 reducesum
1. reduce 1个维度的实现：
```cpp
// load
okk_gdma_32bit_cpy_S2L(tmp_addr, param->input_addr, &shape, &stride, &istride);
// reduceavg
okk_bdc_avg_pool2d(sum_addr, tmp_addr, &shape, shape.h, 1, NULL, NULL);
// mul to sum
okk_bdc_mul_C(sum_addr, sum_addr, shape.h, &sum_shape, NULL, NULL);
// store
okk_gdma_32bit_cpy_L2S(param->output_addr, sum_addr, &sum_shape, NULL, NULL);
```
2. reduce 2个维度的实现：
```cpp
okk_gdma_32bit_cpy_S2L(tmp_addr, param->input_addr, &shape, NULL, NULL);
okk_bdc_avg_pool2d(sum_addr, tmp_addr, &shape, shape.h, shape.w, NULL, NULL);
okk_bdc_mul_C(sum_addr, sum_addr, shape.h * shape.w, &sum_shape, NULL, NULL);
okk_gdma_32bit_cpy_L2S(param->output_addr, sum_addr, &sum_shape, NULL, NULL);
```
3. reduce 3个维度的实现：
```cpp
// 1. reduce h, w ...
// 2. reduce c
shape.c = 1;
shape.h = 1;
shape.w = param->C;
sum_shape.c = 1;
okk_gdma_32bit_cpy_S2L(tmp_addr, param->output_addr, &shape, NULL, NULL);
okk_bdc_avg_pool2d(sum_addr, tmp_addr, &shape, shape.h, shape.w, NULL, NULL);
okk_bdc_mul_C(sum_addr, sum_addr, shape.h * shape.w, &sum_shape, NULL, NULL);
okk_gdma_32bit_cpy_L2S(param->output_addr, sum_addr, &sum_shape, NULL, NULL);
```

### 3.3 rgb2bgr
逐channel拷贝实现如下：
```cpp
int size = param->size;
dim4 shape = { .n = 1, .c = 1, .h = size, .w = 1 };
dim4 istride = { .n = 1, .c = 1, .h = 3, .w = 1 };
dim4 ostride = { .n = 1, .c = 1, .h = 3, .w = 1 };
// copy r
okk_gdma_32bit_cpy_S2S(param->output_addr + 8, param->input_addr, &shape, &ostride, &istride);
// copy g ...
// copy b ...
```

### 3.4 transpose
转置操作根据不同的permute参数来选择不同的stride即可，以下为一种实现：
```cpp
// permute = [0, 2, 1, 3]
int channel = param->C;
int height = param->H;
dim4 shape = { .n = param->N, .c = height, .h = channel, .w = param->W };
dim4 istride = { .n = height * channel * shape.w, .c = shape.w, .h = height * shape.w, .w = 1 };
dim4 ostride = { .n = shape.c * shape.h * shape.w, .c = shape.h * shape.w, .h = shape.w, .w = 1 };
okk_gdma_32bit_cpy_S2S(param->output_addr, param->input_addr, &shape, &ostride, &istride);
```

## 4. 优化分析
针对以上实现，分析优化方法。rgb2bgr和transpose为纯访存操作，优化空间较小；avgpool和reducesum的实现相似，主要的优化思路如下：
1. gdma bdc parallel优化，如batch分块优化后如下：
```cpp
okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr, &shape, NULL, NULL);
okk_parallel_start();
okk_bdc_avg_pool2d(output_addr, input_addr, &shape, 1, 49, NULL, NULL);
okk_gdma_32bit_cpy_S2L(input_addr2, param->input_addr + iblock, &shape, NULL, NULL);
okk_parallel_end();
int i = 1;
for (; i < t - 1; i++) {
    okk_parallel_start();
    okk_gdma_32bit_cpy_S2L(input_addrs[(i+1)%2], param->input_addr + iblock * (i+1), &shape, NULL, NULL);
    okk_bdc_avg_pool2d(output_addrs[i%2], input_addrs[i%2], &shape, 1, 49, NULL, NULL);
    okk_gdma_32bit_cpy_L2S(param->output_addr + oblock * (i - 1), output_addrs[(i-1)%2], &avg_shape, NULL, NULL);
    okk_parallel_end();
}
okk_parallel_start();
okk_bdc_avg_pool2d(output_addrs[i%2], input_addrs[i%2], &shape, 1, 49, NULL, NULL);
okk_gdma_32bit_cpy_L2S(param->output_addr + oblock * (i - 1), output_addrs[(i-1)%2], &avg_shape, NULL, NULL);
okk_parallel_end();
okk_gdma_32bit_cpy_L2S(param->output_addr + oblock * i, output_addrs[i%2], &avg_shape, NULL, NULL);
```
2. 实验验证 `pads != 0 & count_include_pad=false` 的两种实现思路，即针对角和边缘分别采用不同kernel_size的avgpool，和使用相同kernel_size的avgpool后通过
乘常数的方式转换为不计算pad的结果；比较性能差异，选择较快的方案实现；
3. 在乘常量时可以通过stride来合并多次乘法操作为1次，减少指令数目；
