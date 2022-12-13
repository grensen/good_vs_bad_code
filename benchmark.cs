// for more info https://github.com/grensen/good_vs_bad_code
// based on https://github.com/grensen/multi-core
#if DEBUG
System.Console.WriteLine("Debug mode is on, switch to Release mode");
#endif 
using System.Numerics;
using System.Runtime.InteropServices;

System.Action<string> print = System.Console.WriteLine;

print("\nBegin good code benchmark demo\n");

// get data
AutoData d = new(@"C:\mnist\");

// define neural network 
int[] network = { 784, 100, 100, 10 };
var LEARNINGRATE = 0.0005f;
var MOMENTUM = 0.67f;
var EPOCHS = 50;
var BATCHSIZE = 800;
var FACTOR = 0.99f;

RunDemo(network, d, LEARNINGRATE, MOMENTUM, FACTOR, EPOCHS, BATCHSIZE);

print("\nEnd good code benchmark demo");

//+---------------------------------------------------------------------+

static void RunDemo(int[] network, AutoData d, float LEARNINGRATE, float MOMENTUM, float FACTOR, int EPOCHS, int BATCHSIZE)
{
    System.Console.WriteLine("NETWORK      = " + string.Join("-", network));
    System.Console.WriteLine("LEARNINGRATE = " + LEARNINGRATE);
    System.Console.WriteLine("MOMENTUM     = " + MOMENTUM);
    System.Console.WriteLine("BATCHSIZE    = " + BATCHSIZE);
    System.Console.WriteLine("EPOCHS       = " + EPOCHS);
    System.Console.WriteLine("FACTOR       = " + FACTOR);

    var sGoodTime = 0.0f;
    if (!false)
    {
        Net goodSingleNet = new(network);
        System.Console.WriteLine("\nStart good code reproducible Single-Core training");
        sGoodTime = RunNet(false, d, goodSingleNet, 60000, LEARNINGRATE, MOMENTUM, FACTOR, EPOCHS, BATCHSIZE);
    }
    var mGoodTime = 0.0f;
    if (!false)
    {
        Net goodMultiNet = new(network);
        System.Console.WriteLine("\nStart good code non reproducible Multi-Core training");
        mGoodTime = RunNet(true, d, goodMultiNet, 60000, LEARNINGRATE, MOMENTUM, FACTOR, EPOCHS, BATCHSIZE);
    }

    System.Console.WriteLine("\nTotal time good code = " + (sGoodTime + mGoodTime).ToString("F2") + "s"
        + ", reference code = " + (56.13).ToString("F2") + "s");

    System.Console.WriteLine("\nGood code Single time = " + sGoodTime.ToString("F2") + "s" + " good code Multi time = " + mGoodTime.ToString("F2") + "s");

    System.Console.WriteLine("\nGood code Multi-Core was " + (sGoodTime / mGoodTime).ToString("F2") + " times faster than good code Single-Core");

    System.Console.WriteLine("\nGood code was " + (56.13 / (sGoodTime + mGoodTime)).ToString("F2") + " times faster than reference code");
}

static float RunNet(bool multiCore, AutoData d, Net neural, int len, float lr, float mom, float FACTOR, int EPOCHS, int BATCHSIZE)
{
    DateTime elapsed = DateTime.Now;
    RunTraining(multiCore, elapsed, d, neural, len, lr, mom, FACTOR, EPOCHS, BATCHSIZE);
    return RunTest(multiCore, elapsed, d, neural, 10000);

    static void RunTraining(bool multiCore, DateTime elapsed, AutoData d, Net neural, int len, float lr, float mom, float FACTOR, int EPOCHS, int BATCHSIZE)
    {
        float[] delta = new float[neural.weights.Length];
        for (int epoch = 0, B = len / BATCHSIZE; epoch < EPOCHS; epoch++, lr *= FACTOR, mom *= FACTOR)
        {
            bool[] c = new bool[B * BATCHSIZE];
            for (int b = 0; b < B; b++)
            {
                if (multiCore)
                    System.Threading.Tasks.Parallel.ForEach(
                        System.Collections.Concurrent.Partitioner.Create(b * BATCHSIZE, (b + 1) * BATCHSIZE), range =>
                        {
                            for (int x = range.Item1, X = range.Item2; x < X; x++)
                                c[x] = EvalAndTrain(x, d.samplesTraining, neural, delta, d.labelsTraining[x]);
                        });
                else
                    for (int x = b * BATCHSIZE, X = (b + 1) * BATCHSIZE; x < X; x++)
                        c[x] = EvalAndTrain(x, d.samplesTraining, neural, delta, d.labelsTraining[x]);

                Update(neural.weights, delta, lr, mom);
            }
            int correct = c.Count(n => n); // for (int i = 0; i < len; i++) if (c[i]) correct++;
            if ((epoch + 1) % 10 == 0)
                PrintInfo("Epoch = " + (1 + epoch), correct, B * BATCHSIZE, elapsed);
        }
    }
    static float RunTest(bool multiCore, DateTime elapsed, AutoData d, Net neural, int len)
    {
        bool[] c = new bool[len];
        if (multiCore)
            System.Threading.Tasks.Parallel.ForEach(System.Collections.Concurrent.Partitioner.Create(0, len), range =>
            {
                for (int x = range.Item1; x < range.Item2; x++)
                    c[x] = EvalTest(x, d.samplesTest, neural, d.labelsTest[x]);
            });
        else
            for (int x = 0; x < len; x++)
                c[x] = EvalTest(x, d.samplesTest, neural, d.labelsTest[x]);

        int correct = c.Count(n => n); // int correct = 0; //for (int i = 0; i < c.Length; i++) if (c[i]) correct++;
        PrintInfo("Test", correct, 10000, elapsed);
        return (float)((DateTime.Now - elapsed).TotalMilliseconds / 1000.0f);
    }
}
static bool EvalAndTrain(int x, byte[] samples, Net neural, float[] delta, byte t)
{
    float[] neuron = new float[neural.neuronLen];
    int p = Eval(x, neural, samples, neuron);
    if (neuron[neural.neuronLen - neural.net[^1] + t] < 0.99)
        BP(neural, neuron, t, delta);
    return p == t;

    static void BP(Net neural, Span<float> neuron, int target, float[] delta)
    {
        Span<float> gradient = stackalloc float[neuron.Length];

        // output error gradients, hard target as 1 for its class
        for (int r = neuron.Length - neural.net[neural.layers], p = 0; r < neuron.Length; r++, p++)
            gradient[r] = target == p ? 1 - neuron[r] : -neuron[r];
        for (int i = neural.layers - 1, j = neuron.Length - neural.net[neural.layers], k = neuron.Length, m = neural.weights.Length; i >= 0; i--)
        {
            int right = neural.net[i + 1], left = neural.net[i];
            k -= right; j -= left; m -= right * left;

            for (int l = j, w = m; l < left + j; l++, w += right)
            {
                var n = neuron[l];
                if (n > 0)
                {
                    int r = 0; var sum = 0.0f;
                    for (; r < right - 8; r += 8) // 8
                    {
                        int kr = k + r, wr = w + r;
                        var g = gradient[kr]; delta[wr] += n * g; sum += neural.weights[wr] * g;
                        g = gradient[kr + 1]; delta[wr + 1] += n * g; sum += neural.weights[wr + 1] * g;
                        g = gradient[kr + 2]; delta[wr + 2] += n * g; sum += neural.weights[wr + 2] * g;
                        g = gradient[kr + 3]; delta[wr + 3] += n * g; sum += neural.weights[wr + 3] * g;
                        g = gradient[kr + 4]; delta[wr + 4] += n * g; sum += neural.weights[wr + 4] * g;
                        g = gradient[kr + 5]; delta[wr + 5] += n * g; sum += neural.weights[wr + 5] * g;
                        g = gradient[kr + 6]; delta[wr + 6] += n * g; sum += neural.weights[wr + 6] * g;
                        g = gradient[kr + 7]; delta[wr + 7] += n * g; sum += neural.weights[wr + 7] * g;
                    }
                    for (; r < right; r++)
                    {
                        int wr = r + w;
                        var g = gradient[k + r];
                        sum += neural.weights[wr] * g; delta[wr] += n * g;
                    }
                    gradient[l] = sum;
                }
            }
        }
    }
}
static bool EvalTest(int x, byte[] samples, Net neural, byte t)
{
    float[] neuron = new float[neural.neuronLen];
    int p = Eval(x, neural, samples, neuron);
    return p == t;
}
static int Eval(int x, Net neural, byte[] samples, float[] neuron)
{
    FeedInput(x, samples, neuron);
    FeedForward(neuron, neural.weights, neural.net);
    Softmax(neuron, neural.net[neural.layers]);
    return Argmax(neural, neuron);

    static void FeedInput(int x, byte[] samples, Span<float> neuron)
    {
        for (int i = 0, ii = x * 784; i < 784; i += 8)
        {
            var n = samples[ii++];
            if (n > 0) neuron[i] = n / 255f;
            n = samples[ii++];
            if (n > 0) neuron[i + 1] = n / 255f;
            n = samples[ii++];
            if (n > 0) neuron[i + 2] = n / 255f;
            n = samples[ii++];
            if (n > 0) neuron[i + 3] = n / 255f;
            n = samples[ii++];
            if (n > 0) neuron[i + 4] = n / 255f;
            n = samples[ii++];
            if (n > 0) neuron[i + 5] = n / 255f;
            n = samples[ii++];
            if (n > 0) neuron[i + 6] = n / 255f;
            n = samples[ii++];
            if (n > 0) neuron[i + 7] = n / 255f;
        }
    }

    static void FeedForwardAdvancedSpanEachLayer
        (Span<float> neuron, ReadOnlySpan<float> weights, ReadOnlySpan<int> net)
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



    static void FeedForwardDefaultArrayUnrolled
        (float[] neurons, float[] weights, int[] net)
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

    static void FeedForwardDefaultSpanEachLayer
        (Span<float> neurons, ReadOnlySpan<float> weights, ReadOnlySpan<int> net)
    {
        for (int i = 0, j = 0, k = net[0], m = 0; i < net.Length - 1; i++)
        {
            int left = net[i], right = net[i + 1];
            Span<float> localOut = stackalloc float[right];
            for (int l = 0, w = m; l < left; l++, w += right)
            {
                float n = neurons[j + l];
                if (n <= 0) continue;
                ReadOnlySpan<float> localWts = weights.Slice(w, right);
                for (int r = 0; r < localOut.Length; r++)
                    localOut[r] = localWts[r] * n + localOut[r];
            }
            localOut.CopyTo(neurons.Slice(k, right));
            m += left * right; j += left; k += right;
        }
    }
    
    static void FeedForwardDefaultSpanEachInput
    (Span<float> neurons, ReadOnlySpan<float> weights, ReadOnlySpan<int> net)
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

    static void FeedForwardVectorSIMD
        (Span<float> neurons, ReadOnlySpan<float> weights, ReadOnlySpan<int> net)
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

    static void FeedForward//VectorSIMDNoCopy
    (Span<float> neurons, ReadOnlySpan<float> weights, ReadOnlySpan<int> net)
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

    static void FeedForwardDefaultArray
        (float[] neurons, float[] weights, int[] net)
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

    static void FeedForwardNaive
    (float[] neurons, float[] weights, int[] net)
    {
        for (int k = net[0], w = 0, i = 0; i < net.Length - 1; i++)
        {
            int right = net[i + 1];
            for (int l = k - net[i]; l < k; l++, w += right)
            {
                float n = neurons[l];
                if (n <= 0) continue;
                for (int r = 0; r < right; r++)
                    neurons[r + k] += weights[r + w] * n;
            }
            k += right;
        }
    }

    static void FeedForwardGoodCodeOptimized
        (float[] neurons, float[] weights, int[] net)
    {
        for (int k = net[0], w = 0, i = 0; i < net.Length - 1; i++)
        {
            int right = net[i + 1];
            for (int l = k - net[i]; l < k; l++, w += right)
            {
                float n = neurons[l];
                if (n <= 0) continue;
                int r = 0;
                for (; r < right - 8; r += 8)
                {
                    int wr = w + r, kr = k + r;
                    float p = weights[wr] * n; neurons[kr] += p;
                    p = weights[wr + 1] * n; neurons[kr + 1] += p;
                    p = weights[wr + 2] * n; neurons[kr + 2] += p;
                    p = weights[wr + 3] * n; neurons[kr + 3] += p;
                    p = weights[wr + 4] * n; neurons[kr + 4] += p;
                    p = weights[wr + 5] * n; neurons[kr + 5] += p;
                    p = weights[wr + 6] * n; neurons[kr + 6] += p;
                    p = weights[wr + 7] * n; neurons[kr + 7] += p;
                }
                // source loop for (; r < right; r++) { float p = neural.weights[r + w] * n; neuron[r + k] += p; }
                for (; r < right; r++) { float p = weights[r + w] * n; neurons[r + k] += p; }
            }
            k += right;
        }
    }

    static void Softmax(Span<float> neuron, int output)
    {
        float scale = 0;
        for (int n = neuron.Length - output; n < neuron.Length; n++)
            scale += neuron[n] = MathF.Exp(neuron[n]);
        for (int n = neuron.Length - output; n < neuron.Length; n++)
            neuron[n] /= scale;
    }
    static int Argmax(Net neural, Span<float> neuron)
    {
        float max = neuron[neuron.Length - neural.net[neural.layers]];
        int prediction = 0;
        for (int i = 1; i < neural.net[neural.layers]; i++)
        {
            float n = neuron[i + neuron.Length - neural.net[neural.layers]];
            if (n > max) { max = n; prediction = i; } // grab maxout prediction here
        }
        return prediction;
    }
}
static void Update(float[] weight, float[] delta, float lr, float mom)
{
    for (int w = 0; w < weight.Length; w++)
    {
        var d = delta[w] * lr;
        weight[w] += d;
        delta[w] *= mom;
    }
}
static void PrintInfo(string str, int correct, int all, DateTime elapsed)
{
    System.Console.WriteLine(str + " accuracy = " + (correct * 100.0 / all).ToString("F2")
        + " correct = " + correct + "/" + all + " time = " + ((DateTime.Now - elapsed).TotalMilliseconds / 1000.0).ToString("F2") + "s");
}

struct Net
{
    public int[] net;
    public int neuronLen, layers;
    public float[] weights;
    public Net(int[] net)
    {
        this.net = net;
        this.neuronLen = net.Sum();
        this.layers = net.Length - 1;
        this.weights = Glorot(this.net);
        static float[] Glorot(int[] net)
        {
            int len = 0;
            for (int n = 0; n < net.Length - 1; n++)
                len += net[n] * net[n + 1];

            float[] weights = new float[len];
            Erratic rnd = new(12345);

            for (int i = 0, w = 0; i < net.Length - 1; i++, w += net[i - 0] * net[i - 1]) // layer
            {
                float sd = MathF.Sqrt(6.0f / (net[i] + net[i + 1]));
                for (int m = w; m < w + net[i] * net[i + 1]; m++) // weights
                    weights[m] = rnd.NextFloat(-sd * 1.0f, sd * 1.0f);
            }
            return weights;
        }
    }
}
struct AutoData // https://github.com/grensen/easy_regression#autodata
{
    public string source;
    public byte[] samplesTest, labelsTest;
    public byte[] samplesTraining, labelsTraining;
    public AutoData(string yourPath)
    {
        this.source = yourPath;

        // hardcoded urls from my github
        string trainDataUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/train-images.idx3-ubyte";
        string trainLabelUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/train-labels.idx1-ubyte";
        string testDataUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/t10k-images.idx3-ubyte";
        string testnLabelUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/t10k-labels.idx1-ubyte";

        // change easy names 
        string d1 = @"trainData", d2 = @"trainLabel", d3 = @"testData", d4 = @"testLabel";

        if (!File.Exists(yourPath + d1)
            || !File.Exists(yourPath + d2)
              || !File.Exists(yourPath + d3)
                || !File.Exists(yourPath + d4))
        {
            System.Console.WriteLine("Data does not exist");
            if (!Directory.Exists(yourPath)) Directory.CreateDirectory(yourPath);

            // padding bits: data = 16, labels = 8
            System.Console.WriteLine("Download MNIST dataset from GitHub");
            this.samplesTraining = (new System.Net.WebClient().DownloadData(trainDataUrl)).Skip(16).Take(60000 * 784).ToArray();
            this.labelsTraining = (new System.Net.WebClient().DownloadData(trainLabelUrl)).Skip(8).Take(60000).ToArray();
            this.samplesTest = (new System.Net.WebClient().DownloadData(testDataUrl)).Skip(16).Take(10000 * 784).ToArray();
            this.labelsTest = (new System.Net.WebClient().DownloadData(testnLabelUrl)).Skip(8).Take(10000).ToArray();

            System.Console.WriteLine("Save cleaned MNIST data into folder " + yourPath + "\n");
            File.WriteAllBytes(yourPath + d1, this.samplesTraining);
            File.WriteAllBytes(yourPath + d2, this.labelsTraining);
            File.WriteAllBytes(yourPath + d3, this.samplesTest);
            File.WriteAllBytes(yourPath + d4, this.labelsTest); return;
        }
        // data on the system, just load from yourPath
        System.Console.WriteLine("Load MNIST data and labels from " + yourPath + "\n");
        this.samplesTraining = File.ReadAllBytes(yourPath + d1).Take(60000 * 784).ToArray();
        this.labelsTraining = File.ReadAllBytes(yourPath + d2).Take(60000).ToArray();
        this.samplesTest = File.ReadAllBytes(yourPath + d3).Take(10000 * 784).ToArray();
        this.labelsTest = File.ReadAllBytes(yourPath + d4).Take(10000).ToArray();
    }
}
class Erratic // https://jamesmccaffrey.wordpress.com/2019/05/20/a-pseudo-pseudo-random-number-generator/
{
    private float seed;
    public Erratic(float seed2)
    {
        this.seed = this.seed + 0.5f + seed2;  // avoid 0
    }
    public float Next()
    {
        var x = Math.Sin(this.seed) * 1000;
        var result = (float)(x - Math.Floor(x));  // [0.0,1.0)
        this.seed = result;  // for next call
        return this.seed;
    }
    public float NextFloat(float lo, float hi)
    {
        var x = this.Next();
        return (hi - lo) * x + lo;
    }
};