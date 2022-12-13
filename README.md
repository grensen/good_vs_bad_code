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

## Default Implementation With Arrays
~~~cs
static void FeedForwardDefaultArray(float[] neurons, float[] weights, int[] net)
{
    for (int i = 0, j = 0, k = net[0], m = 0; i < net.Length - 1; i++)
    {
        int left = net[i], right = net[i + 1];
        for (int l = 0, w = m; l < left; l++)
        {
            float n = neurons[j + l];
            if (n > 0)
                for (int r = 0; r < right; r++)
                    neurons[k + r] += n * weights[w + r];
            w += right;
        }
        m += left * right; j += left; k += right;
    }
}
~~~

## Default Arrays Unrolled
~~~cs
static void FeedForwardDefaultArrayUnrolled(float[] neurons, float[] weights, int[] net)
{
    for (int i = 0, j = 0, k = net[0], m = 0; i < net.Length - 1; i++)
    {
        int left = net[i], right = net[i + 1];
        for (int l = 0, w = m; l < left; l++)
        {
            float n = neurons[j + l];
            if (n > 0)
            {
                int r = 0;
                for (; r < right - 8; r = 8 + r)
                {
                    neurons[k + r] = n * weights[w + r] + neurons[k + r];
                    neurons[k + r + 1] = n * weights[w + r + 1] + neurons[k + r + 1]; 
                    neurons[k + r + 2] = n * weights[w + r + 2] + neurons[k + r + 2];
                    neurons[k + r + 3] = n * weights[w + r + 3] + neurons[k + r + 3]; 
                    neurons[k + r + 4] = n * weights[w + r + 4] + neurons[k + r + 4];
                    neurons[k + r + 5] = n * weights[w + r + 5] + neurons[k + r + 5]; 
                    neurons[k + r + 6] = n * weights[w + r + 6] + neurons[k + r + 6];
                    neurons[k + r + 7] = n * weights[w + r + 7] + neurons[k + r + 7];
                }
                for (; r < right; r++)
                    neurons[k + r] += n * weights[w + r];
            }
            w += right;
        }
        m += left * right; j += left; k += right;
    }
}
~~~

## Default Implementation With Spans
~~~cs
static void FeedForwardDefaultSpanEachInput(Span<float> neurons, ReadOnlySpan<float> weights, ReadOnlySpan<int> net)
{
    for (int i = 0, j = 0, k = net[0], m = 0; i < net.Length - 1; i++)
    {
        int left = net[i], right = net[i + 1];
        for (int l = 0, w = m; l < left; l++, w += right)
        {
            float n = neurons[j + l];
            if (n <= 0) continue;
            ReadOnlySpan<float> localWts = weights.Slice(w, right);
            Span<float> localOut = neurons.Slice(k, right);
            for (int r = 0; r < localOut.Length; r++)
                localOut[r] = localWts[r] * n + localOut[r];               
        }
        m += left * right; j += left; k += right;
    }
}
~~~

## Advanced Spans Layer Wise
~~~cs
static void FeedForwardAdvancedSpanEachLayer(Span<float> neuron, ReadOnlySpan<float> weights, ReadOnlySpan<int> net)
{
    for (int k = net[0], w = 0, i = 0; i < net.Length - 1; i++)
    {
        Span<float> activations = neuron.Slice(k, net[i + 1]);
        ReadOnlySpan<float> localInp = neuron.Slice(k - net[i], net[i]);
        Span<float> localOut = stackalloc float[net[i + 1]];
        for (int l = 0; l < net[i]; w = w + localOut.Length, l++)
        {
            float n = localInp[l];
            if (n <= 0) continue;
            ReadOnlySpan<float> wts = weights.Slice(w, localOut.Length);
            for (int r = 0; r < localOut.Length; r++)
                localOut[r] = wts[r] * n + localOut[r];
        }
        k = localOut.Length + k;
        localOut.CopyTo(activations);
    }
}
~~~

## Advanced Vector SIMD
~~~cs
static void FeedForwardVectorSIMD(Span<float> neurons, ReadOnlySpan<float> weights, ReadOnlySpan<int> net)
{
    for (int k = net[0], w = 0, i = 0; i < net.Length - 1; i++)
    {
        ReadOnlySpan<float> localInp = neurons.Slice(k - net[i], net[i]);
        Span<float> localOut = stackalloc float[net[i + 1]];
        for (int l = 0; l < net[i]; w = localOut.Length + w, l++)
        {
            float n = localInp[l];
            if (n <= 0) continue;
            ReadOnlySpan<float> wts = weights.Slice(w, localOut.Length);
            int r = 0;
            for (; r < localOut.Length - Vector<float>.Count; r += Vector<float>.Count)
            {
                Vector<float> va = new Vector<float>(localOut.Slice(r, Vector<float>.Count));
                Vector<float> vb = new Vector<float>(wts.Slice(r, Vector<float>.Count));
                va += vb * n;
                va.CopyTo(localOut.Slice(r, Vector<float>.Count));
            }
            for (; r < localOut.Length; ++r)
                localOut[r] = wts[r] * n + localOut[r];
        }
        localOut.CopyTo(neurons.Slice(k, net[i + 1]));
        k = localOut.Length + k;
    }
}
~~~

## Advanced Vector SIMD No Copy
~~~cs
static void FeedForwardVectorSIMDNoCopy(Span<float> neurons, ReadOnlySpan<float> weights, ReadOnlySpan<int> net)
{
    for (int k = net[0], w = 0, i = 0; i < net.Length - 1; i++)
    {
        ReadOnlySpan<float> localInp = neurons.Slice(k - net[i], net[i]);
        Span<float> localOut = stackalloc float[net[i + 1]];                         
        for (int l = 0; l < localInp.Length; w = w + localOut.Length, l++)
        {
            float n = localInp[l];
            if (n <= 0) continue;
            ReadOnlySpan<float> wts = weights.Slice(w, localOut.Length);
            ReadOnlySpan<Vector<float>> wtsVecArray = MemoryMarshal.Cast<float, Vector<float>>(wts);
            Span<Vector<float>> resultsVecArray = MemoryMarshal.Cast<float, Vector<float>>(localOut);
            for (int v = 0; v < resultsVecArray.Length; v++)
                resultsVecArray[v] = wtsVecArray[v] * n + resultsVecArray[v];
            for (int r = wtsVecArray.Length * Vector<float>.Count; r < localOut.Length; r++)
                localOut[r] = wts[r] * n + localOut[r];
        }
        Span<float> activations = neurons.Slice(k, localOut.Length);
        localOut.CopyTo(activations);
        k = localOut.Length + k;
    }
}
~~~

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

