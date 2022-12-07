# Good Code vs. Bad Code Demo Using C#


## The Idea
<p align="center">
  <img src="https://github.com/grensen/good_vs_bad_code/blob/main/figures/idea_vs_computer.png?raw=true">
</p>

## The Demo
<p align="center">
  <img src="https://github.com/grensen/good_vs_bad_code/blob/main/figures/good_vs_bad_demo.png?raw=true">
</p>

## The Demo With .NET 7
<p align="center">
  <img src="https://github.com/grensen/good_vs_bad_code/blob/main/figures/good_vs_bad_demo_dotnet7.png?raw=true">
</p>

## How To Unroll a Loop
~~~cs
int n = 100;
int i = 0;
if (unrolling)
    for (; i < n - 4; i += 4)
    {
        sum += data[i + 0];
        sum += data[i + 1];
        sum += data[i + 2];
        sum += data[i + 3];
    }
for (; i < n; i++)
    sum += data[i];
~~~

## Implementation Details That Can Count
~~~cs
// a first?
float[] a;
int b;
~~~

~~~cs
// or b first? 
int b;
float[] a;
~~~

## Visual Studio Profiler
<p align="center">
  <img src="https://github.com/grensen/good_vs_bad_code/blob/main/figures/vs_profiler_modules.png?raw=true">
</p>
https://www.youtube.com/watch?v=y4HV5m5GR7o

## Feed Forward Cost Details
<p align="center">
  <img src="https://github.com/grensen/good_vs_bad_code/blob/main/figures/vs_profiler_caller_info.png?raw=true">
</p>


## Array vs. Span Assembly On sharplab.io
~~~cs
static void SumProductToSpan(Span<float> neurons, Span<float> weights, float n)
{    
    for(int r = 0; r < neurons.Length; r++)
    {                 
       neurons[r] = neurons[r] + weights[r] * n;
    }   
}

static void SumProductToArray(float[] neurons, float[] weights, float n)
{    
    for(int r = 0; r < neurons.Length; r++)
    {                 
       neurons[r] = neurons[r] + weights[r] * n;
    }   
}
~~~
https://sharplab.io/

## Assembly Instructions Array
<p align="center">
  <img src="https://github.com/grensen/good_vs_bad_code/blob/main/figures/asm_array.png?raw=true">
</p>

## Assembly Instructions Span
<p align="center">
  <img src="https://github.com/grensen/good_vs_bad_code/blob/main/figures/asm_span.png?raw=true">
</p>

## Floating Point "Features"
<p align="center">
  <img src="https://github.com/grensen/good_vs_bad_code/blob/main/figures/single_vs_one_piece.png?raw=true">
</p>

## SIMD
<p align="center">
  <img src="https://github.com/grensen/good_vs_bad_code/blob/main/figures/simd_bench.png?raw=true">
</p>
https://devblogs.microsoft.com/dotnet/hardware-intrinsics-in-net-core/

## Language Efficiency
<p align="center">
  <img src="https://github.com/grensen/good_vs_bad_code/blob/main/figures/language_efficiency.png?raw=true">
</p>
https://greenlab.di.uminho.pt/wp-content/uploads/2017/09/paperSLE.pdf

## Numeric performance in C, C# and Java
<p align="center">
  <img src="https://github.com/grensen/good_vs_bad_code/blob/main/figures/c_cs_java_numeric_perf.png?raw=true">
</p>
old paper: https://www.itu.dk/people/sestoft/papers/numericperformance.pdf

## EFFICIENT MATRIX MULTIPLICATION USING HARDWAREINTRINSICS AND PARALLELISM WITH C#
<p align="center">
  <img src="https://github.com/grensen/good_vs_bad_code/blob/main/figures/efficient_matrix_multiplication.png?raw=true">
</p>
https://docplayer.net/227695722-Efficient-matrix-multiplication-using-hardware-intrinsics-and-parallelism-with-c.html

