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

## Forward Propagation Default Implementation With Arrays
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

## Backpropagation Default
~~~cs
static void Backprop(int[] net, float[] weights, float[] neuron, float[] delta, int target)
{
    Span<float> gradient = stackalloc float[neuron.Length];

    for (int r = neuron.Length - net[^1], p = 0; r < neuron.Length; r++, p++)
        gradient[r] = target == p ? 1 - neuron[r] : -neuron[r];

    for (int i = net.Length - 2, j = neuron.Length - net[^1], k = neuron.Length, m = weights.Length; i >= 0; i--)
    {
        int right = net[i + 1], left = net[i];
        k -= right; j -= left; m -= right * left;

        for (int l = j, w = m; l < left + j; l++, w += right)
        {
            var n = neuron[l];
            if (n > 0)
            {
                float sum = 0.0f;
                for (int r = 0; r < right; r++)
                {
                    int wr = r + w;
                    var g = gradient[k + r];
                    sum += weights[wr] * g; delta[wr] += n * g;
                }
                gradient[l] = sum;
            }
        }
    }
}
~~~

## Backpropagation SIMD No Copy
~~~cs
static void BackpropSIMDNoCopy(int[] net, Span<float> weights, float[] neuron, Span<float> delta, int target)
{
    Span<float> gradient = stackalloc float[neuron.Length];

    // output error gradients, hard target as 1 for its class
    for (int r = neuron.Length - net[^1], p = 0; r < neuron.Length; r++, p++)
        gradient[r] = target == p ? 1 - neuron[r] : -neuron[r];

    for (int j = neuron.Length - net[^1], k = neuron.Length, m = weights.Length, i = net.Length - 2; i >= 0; i--)
    {
        int right = net[i + 1], left = net[i];
        k -= right; j -= left; m -= right * left;
        Span<float> gra = gradient.Slice(k, right);

        for (int l = 0, w = m; l < left; l++, w += right)
        {
            var n = neuron[l + j];
            if (n <= 0) continue;

            Span<float> wts = weights.Slice(w, right);
            Span<float> dts = delta.Slice(w, right);

            Span<Vector<float>> graVec = MemoryMarshal.Cast<float, Vector<float>>(gra);
            Span<Vector<float>> dtsVec = MemoryMarshal.Cast<float, Vector<float>>(dts);
            Span<Vector<float>> wtsVec = MemoryMarshal.Cast<float, Vector<float>>(wts);

            var sumVec = Vector<float>.Zero;  
            for (int v = 0; v < wtsVec.Length; v++)
            {
                var gVec = graVec[v];
                sumVec = wtsVec[v] * gVec + sumVec;
                dtsVec[v] = n * gVec + dtsVec[v];
            }

            // changed float result with vector sum
            float sum = Vector.Sum(sumVec);

            for (int r = wtsVec.Length * Vector<float>.Count; r < wts.Length; r++)
            {
                var g = gra[r];
                sum = wts[r] * g + sum;
                dts[r] = n * g + dts[r];
            }
            gradient[l + j] = sum;
        }
    }
}
~~~


## Update Weights Default
~~~cs
static void UpdateDefault(float[] weight, float[] delta, float lr, float mom)
{
    for (int w = 0; w < weight.Length; w++)
    {
        var d = delta[w] * lr;
        weight[w] += d;
        delta[w] *= mom;
    }  
}
~~~

## Update Weights SIMD
~~~cs
static void UpdateSIMDNoCopy(float[] weight, float[] delta, float lr, float mom)
{
    Span<Vector<float>> weightVecArray = MemoryMarshal.Cast<float, Vector<float>>(weight);
    Span<Vector<float>> deltaVecArray = MemoryMarshal.Cast<float, Vector<float>>(delta);
    for (int v = 0; v < weightVecArray.Length; v++)
    {
        weightVecArray[v] += deltaVecArray[v] * lr;
        deltaVecArray[v] *= mom;
    }
    for (int w = weightVecArray.Length * Vector<float>.Count; w < weight.Length; w++)
    {
        weight[w] += delta[w] * lr;
        delta[w] *= mom;
    }
}
~~~

## Update ChatGPT Version
~~~cs
static void UpdateChatGPT(float[] weight, float[] delta, float lr, float mom)
{
    // Pre-compute the values of lr and mom,
    // and use local variables to store these values.
    float lr_value = lr;
    float mom_value = mom;

    // Use the "unsafe" keyword to enable pointer arithmetic.
    unsafe
    {
        // Use the "fixed" keyword to fix the arrays
        // in memory and get a pointer to their elements.
        fixed (float* w = weight, d = delta)
        {
            // Iterate over the elements in the arrays
            // using pointer arithmetic.
            for (int i = 0; i < weight.Length; i++)
            {
                // Use local variables to store the values
                // of w[i] and d[i], and update these values
                // using the += and *= operators.
                float w_value = w[i];
                float d_value = d[i];
                w_value += d_value * lr_value;
                d_value *= mom_value;

                // Update the values in the arrays using
                // the pointer and the dereference operator (*).
                *(w + i) = w_value;
                *(d + i) = d_value;
            }
        }
    }
}
~~~


## ChatGPT With Hard Prompts
~~~cs
static void Update//ChatGPTUnrolledAVX2
    (float[] weight, float[] delta, float lr, float mom)
{
    unsafe
    {
        fixed (float* w = weight, d = delta)
        {
            int i = 0;
            for (; i < weight.Length - 7; i += 8)
            {
                // Load 8 floats from the weight and delta arrays.
                var wVector = Avx2.LoadVector256(w + i);
                var dVector = Avx2.LoadVector256(d + i);

                // Convert the learning rate and momentum factors to vectors.
                var lrVector = Avx2.BroadcastScalarToVector256(&lr);
                var momVector = Avx2.BroadcastScalarToVector256(&mom);

                // Update the weight and delta vectors using AVX2 instructions.
                wVector += Avx2.Multiply(dVector, lrVector);
                dVector *= momVector;

                // Store the updated weight and delta vectors back to memory.
                Avx2.Store(w + i, wVector);
                Avx2.Store(d + i, dVector);
            }
            // Update the remaining elements using a regular for loop.
            for (; i < weight.Length; i++)
            {
                w[i] += d[i] * lr;
                d[i] *= mom;
            }
        }
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

