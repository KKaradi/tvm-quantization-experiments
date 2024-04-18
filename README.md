# Basic Quantization Results
The following is an outline of the setup of the basic quantization experiemnt.

The source code for the benchmarking can be seen in the `quantize_benchmark.py`. The code first pulls a instance of a ResNet50 model implemented in PyTorch then converts it into an executable function. As my machine only supports CPU, the code compiles to an executable using LLVM to run on the CPU. Then the code with perform various quantizations of the model, with the source code for each being written to `./source_code` and benchmarked for performance and written to `benchmarks.txt`

### Here is an list of each of the different quanitzation configurations tested.

>
> - nbit 16 with dtype int16
> - nbit 8 with dtype int8
> - nbit 4 with dtype int8
> - nbit 2 with dtype int8
> - nbit 1 with dtype int8
> - **nbit 4 with dtype int4**
> - **nbit 2 with dtype int2**
> - **nbit 1 with dtype int1**


Of the following list all were successful in compliing with their respective source code being seen in the `./source_code` except the last 3 bolded which failed with the follwing errors.

- InternalError: Check failed: value_dtype.bits() >= 8 (4 vs. 8): 
- InternalError: Check failed: data_bits % 8 == 0U (2 vs. 0) : Need to load/store by multiple of bytes
- InternalError: Check failed: (value == 0 || value == 1) is false: ValueError: -1 exceeds range of int1

These are expected as our CPU won't natively support such small word widths with the smallest being 8bits which we see do see compile properly. 

## Source Code Analysis

By diffing the `no_quantization` source code and the `nbit8_dtypeint8` we can see that indeed the types in the compiled code have been changed from type float to type i8 as expected. Thus our quantization is being successfully applied to the compiled code.

**UNQUANTIZED**
> %closure_loop_parallel_ax0.ax1.fused.ax2.fused = type { float*, float* } 

**QUANTIZED**
> %closure_loop_parallel_ax0.ax1.fused.ax2.fused = type { i8*, i8*, i8* }

When we examine the difference between the `nbit8_dtypeint8` and `nbit4_dtypeint8` our results get less clear.

When looking at the diff provided in `diff_int8_vs_int4.txt` we can see that in some parts of the compiled code int4 has returned to using `float` type where int8 used `i8`, where as in other times the two implementations seem to use very similar types. It seems that changing the nbit without the datatype has these unintented effects and it isn't clear how the nbit is changing the compiled code. When reading TVM's source code we see this field for nbits commented as `Number of bit for every kind of annotate field.` yet we can see that this isn't an incredibly clear process. It is possible that these annoted fields change the implementation of algorithms but not the typing as the dtypes are still using int8.

When we examine the difference between the `nbit8_dtypeint8` and `nbit1_dtypeint8` we again see some modifications to the source code.

**int8**
> %closure_loop_parallel_ax0.ax1.fused.ax2.fused = type { i8*, i8*, i8* }

**int1**
> %closure_loop_parallel_ax0.ax1.fused.ax2.fused = type { i8* }

This time it does seem like the int1 is being considered and resulting in different source code as using only one int8 in the decleration which does show a decrease in total bits allocated. That said it I was curious why this was only seen with the nbit = 1 while not for 2 or 4. I'd like to investigate this further.

We can also see a far clear difference when we diff the `nbit16_dtypeint16` and `nbit8_dtypeint8`. After diffing we see

**int16**
> %closure_loop_parallel_ax0.ax1.fused.ax2.outer.fused = type { i16*, i16*, i16* }

**int8**
> %closure_loop_parallel_ax0.ax1.fused.ax2.fused.39 = type { i8*, i8*, i8* }

Here its very clear that each is using there respective data type because both int8 and int16 are both word lengths natively supported on the architecture of my computer. For these reasons I expect these to be the best examples of proper quantization.

## Runtime Analysis

Here we get incredibly interesting results when we look at the `benchmark.txt`. These results are also written here for your convenience.

```
no_quantization
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
  123.2618     123.8307     131.2479     113.6455      4.2161                  
nbit8_dtypeint8
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
  498.1155     496.3836     505.6944     492.0317      4.9936                  
nbit4_dtypeint8
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
  495.4293     496.4415     503.6331     480.5381      7.3335                  
nbit2_dtypeint8
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
  512.3324     513.7891     520.5761     500.0497      7.1854                  
nbit16_dtypeint16
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
  295.0180     295.8550     311.1781     278.5558     11.0452                  
nbit1_dtypeint8
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
   5.2328       5.3807       6.1950       2.9750       0.8713                  
```             

At first these results are quite suprising and we find that for one we get the opposite effect that we may have expected generally for the amount of quantization and performance. The best performed models were the least quantized ones with float > int16 > int8 (nbit = 8,4,2). 

An explination for this could potentially be the error message `One or more operators have not been tuned. Please tune your model for better performance.` Looking at the documentation we can see that tvm has the ability to tune parameters to best work on hardware, some examples being the input shapes of tensors. Given that we haven't given the quantized versions the ability to tune, it may be that the before tuning the float and int16 simply performe better as they are more attuned to the specific hardware of my local machine. I'd be interested in tying this on a CUDA GPU device to see if this same behavior was seen even before optimization. In may be very well the case that more expected results are seen as a the differences in GPU and CPU architecture but without having access to a GPU device with CUDA myself I would not be able to tell. 

That said one speed up that was observed was for the nbit1_dtypeint8 where it ran in roughly 5ms, far faster than any of the others. This is somewhat unexpected but when looking at how the source code was modified we saw that the nbit 1 option lead to far bigger optimizations in the compiled code which could explain the faster times. This time is also far faster than any of the other implementations by orders of magnitude. This may be due to the simplicity of a nbit = 1 since there are simply only 1 bit to modify but given the trend of slow nbit=4 and nbit=2 it seems that this may not tell the full picture. I'd also like to investigate this further, finding out why the nbit = 1 acts so strange both in runtime and compilation.

Another very interesting trend is the relationship between int16 and int8. We know that these compiled properly when looking at the source code as both dtypes are supported word lengths on our processor. Yet the 8 bit took nearly twice as long as the 16 bit code. This seems to be counter to what we expect as we expect 8 bits to be more protable and thus faster. Yet considering I was running a 64 bit machine and the extra processor instructions required to mask, shift and work with smaller word lengths it may very well be the case that smaller word lengths are infact hindering the performance of the machine. This could also explain why the floats did far better than even the 16 bit words. Given the fact that floats have their own custom hardware on my machine, they can easily be operated on without the need for masking that even the 16 bit operations would require as the processor is 64 bit. These issues are most likely the things that tuning could resolve, reducing the number of shifts and masks required for the smaller bit operations by changing the input and shapes of the tensors that are passed around allowing these to be done faster and perhaps in parralell leading to the speed ups that we would like to get out of the quantized results. 

## Future

In the future I'd like to investigate all the strange impacts I discovered in this experiment. I'd like to investigate the effect of the one bit size, and try the process of tuning to see how this would effect the runtime speed and the compilation. I'd also like to run this on a CUDA GPU machine to see how this would effect speeds.
