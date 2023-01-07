// https://github.com/grensen/good_vs_bad_code

using System.Numerics;
using System.Runtime.InteropServices;

#if DEBUG
System.Console.WriteLine("Debug mode is on, switch to Release mode");
#endif 

Console.WriteLine("\nBegin benchmark .Net 7 demo\n");

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

Console.WriteLine("\nEnd benchmark .Net 7 demo");

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
    if (false)
    {
        Net goodSingleNet = new(network);
        System.Console.WriteLine("\nStart reproducible Single-Core training");
        sGoodTime = RunNet(false, d, goodSingleNet, 60000, LEARNINGRATE, MOMENTUM, FACTOR, EPOCHS, BATCHSIZE);
    }
    var mGoodTime = 0.0f;
    if (!false)
    {
        Net goodMultiNet = new(network);
        System.Console.WriteLine("\nStart non reproducible Multi-Core training");
        mGoodTime = RunNet(true, d, goodMultiNet, 60000, LEARNINGRATE, MOMENTUM, FACTOR, EPOCHS, BATCHSIZE);
    }
    System.Console.WriteLine("\nTotal time good code = " + (sGoodTime + mGoodTime).ToString("F2") + "s");

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
                                    c[x] = EvalAndTrain(x, d.samplesTrainingF, neural, delta, d.labelsTraining[x]);
                            });
                    else
                        for (int x = b * BATCHSIZE, X = (b + 1) * BATCHSIZE; x < X; x++)
                            c[x] = EvalAndTrain(x, d.samplesTrainingF, neural, delta, d.labelsTraining[x]);

                    System.Threading.Tasks.Parallel.ForEach(
                        System.Collections.Concurrent.Partitioner.Create(0, neural.weights.Length), range =>
                        {
                            Update(range.Item1, range.Item2, neural.weights, delta, lr, mom);
                        });
                }
                int correct = c.Count(n => n); // for (int i = 0; i < len; i++) if (c[i]) correct++;
                if ((epoch + 1) % 10 == 0)
                    PrintInfo("Epoch = " + (1 + epoch), correct, B * BATCHSIZE, elapsed);
            }

            static bool EvalAndTrain(int x, float[] samples, Net neural, float[] delta, byte t)
            {
                float[] neuron = new float[neural.neuronLen];
                int p = Eval(x, neural, samples, neuron);
                if (neuron[neural.neuronLen - neural.net[^1] + t] < 0.99)
                    Backprop(neural.net, neural.weights, neuron, delta, t);
                return p == t;

                static void Backprop(int[] net, Span<float> weights, float[] neuron, Span<float> delta, int target)
                {

                    Span<float> gradient = stackalloc float[net[^1]];

                    // output error gradients, hard target as 1 for its class
                    for (int r = neuron.Length - net[^1], p = 0; r < neuron.Length; r++, p++)
                        gradient[p] = target == p ? 1 - neuron[r] : -neuron[r];

                    for (int j = neuron.Length - net[^1], k = neuron.Length, m = weights.Length, i = net.Length - 2; i >= 0; i--)
                    {
                        int right = net[i + 1], left = net[i];
                        k -= right; j -= left; m -= right * left;

                        Span<float> localGradient = stackalloc float[left];
                        for (int l = 0, w = m; l < left; l++, w += right)
                        {
                            var n = neuron[l + j];
                            if (n <= 0) continue;

                            Span<float> wts = weights.Slice(w, right);
                            Span<float> dts = delta.Slice(w, right);

                            Span<Vector<float>> dtsVec = MemoryMarshal.Cast<float, Vector<float>>(dts);
                            Span<Vector<float>> wtsVec = MemoryMarshal.Cast<float, Vector<float>>(wts);
                            Span<Vector<float>> graVec = MemoryMarshal.Cast<float, Vector<float>>(gradient);
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
                                var g = gradient[r];
                                sum = wts[r] * g + sum;
                                dts[r] = n * g + dts[r];
                            }
                            localGradient[l] = sum;
                        }
                        gradient = localGradient;
                    }
                }
            }
            static void Update(int st, int en, Span<float> weights, Span<float> delta, float lr, float mom)
            {
                var weightVecArray = MemoryMarshal.Cast<float, Vector<float>>(weights.Slice(st, en - st));
                var deltaVecArray = MemoryMarshal.Cast<float, Vector<float>>(delta.Slice(st, en - st));
                for (int v = 0; v < weightVecArray.Length; v++)
                {
                    weightVecArray[v] += deltaVecArray[v] * lr;
                    deltaVecArray[v] *= mom;
                }
                for (int w = weightVecArray.Length * Vector<float>.Count + st; w < en; w++)
                {
                    weights[w] += delta[w] * lr;
                    delta[w] *= mom;
                }
            }
        }

        static float RunTest(bool multiCore, DateTime elapsed, AutoData d, Net neural, int len)
        {
            bool[] c = new bool[len];
            if (multiCore)
                System.Threading.Tasks.Parallel.ForEach(System.Collections.Concurrent.Partitioner.Create(0, len), range =>
                {
                    for (int x = range.Item1; x < range.Item2; x++)
                        c[x] = EvalTest(x, d.samplesTestF, neural, d.labelsTest[x]);
                });
            else
                for (int x = 0; x < len; x++)
                    c[x] = EvalTest(x, d.samplesTestF, neural, d.labelsTest[x]);

            int correct = c.Count(n => n); // int correct = 0; //for (int i = 0; i < c.Length; i++) if (c[i]) correct++;
            PrintInfo("Test", correct, 10000, elapsed);
            return (float)((DateTime.Now - elapsed).TotalMilliseconds / 1000.0f);
            static bool EvalTest(int x, float[] samples, Net neural, byte t)
            {
                float[] neuron = new float[neural.neuronLen];
                int p = Eval(x, neural, samples, neuron);
                return p == t;
            }
        }

        static int Eval(int x, Net neural, float[] samples, float[] neuron)
        {
            FeedInput(x, samples, neuron);
            FeedForward(neuron, neural.weights, neural.net);

            var outputsSpan = neuron.AsSpan().Slice(neuron.Length - neural.net[^1], neural.net[^1]);
            Softmax(outputsSpan);
            return ArgmaxSpan(outputsSpan);

            static void FeedInput(int x, Span<float> samples, Span<float> neuron)
            {
                Span<Vector<float>> neuronVec = MemoryMarshal.Cast<float, Vector<float>>(neuron.Slice(0, 784));
                Span<Vector<float>> samplesVec = MemoryMarshal.Cast<float, Vector<float>>(samples.Slice(x * 784, 784));
                for (int i = 0; i < samplesVec.Length; i++)
                    neuronVec[i] = samplesVec[i];
            }
            static void FeedForward(Span<float> neurons, ReadOnlySpan<float> weights, ReadOnlySpan<int> net)
            {
                for (int k = net[0], w = 0, i = 0; i < net.Length - 1; i++)
                {
                    Span<float> localOut = stackalloc float[net[i + 1]];
                    ReadOnlySpan<float> localInp = neurons.Slice(k - net[i], net[i]);
                    for (int l = 0; l < localInp.Length; w = w + localOut.Length, l++)
                    {
                        float n = localInp[l];
                        if (n <= 0) continue;
                        ReadOnlySpan<float> wts = weights.Slice(w, localOut.Length);
                        ReadOnlySpan<Vector<float>> wtsVec = MemoryMarshal.Cast<float, Vector<float>>(wts);
                        Span<Vector<float>> resultsVec = MemoryMarshal.Cast<float, Vector<float>>(localOut);
                        for (int v = 0; v < resultsVec.Length; v++)
                            resultsVec[v] = wtsVec[v] * n + resultsVec[v];
                        for (int r = wtsVec.Length * Vector<float>.Count; r < localOut.Length; r++)
                            localOut[r] = wts[r] * n + localOut[r];
                    }
                    localOut.CopyTo(neurons.Slice(k, localOut.Length));
                    k = localOut.Length + k;
                }
            }
            static void Softmax(Span<float> neuron)
            {
                float scale = 0;
                for (int n = 0; n < neuron.Length; n++) scale += neuron[n] = MathF.Exp(neuron[n]);
                for (int n = 0; n < neuron.Length; n++) neuron[n] /= scale;
            }
            static int ArgmaxSpan(Span<float> neuron)
            {
                int id = 0;
                float max = neuron[0];
                for (int i = 1; i < neuron.Length; i++)
                {
                    if (neuron[i] > max)
                    {
                        max = neuron[i];
                        id = i;
                    }
                }
                return id;
            }
        }

        static void PrintInfo(string str, int correct, int all, DateTime elapsed)
        {
            System.Console.WriteLine(str + " accuracy = " + (correct * 100.0 / all).ToString("F2")
                + " correct = " + correct + "/" + all + " time = " + ((DateTime.Now - elapsed).TotalMilliseconds / 1000.0).ToString("F2") + "s");
        }
    }
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
struct AutoData
{
    public string source;
    public byte[] labelsTraining, labelsTest;
    public float[] samplesTrainingF, samplesTestF;

    public float[] NormalizeData(int n, byte[] samples)
    {
        float[] samplesF = new float[samples.Length];
        for (int i = 0; i < samples.Length; i++)
            samplesF[i] = samples[i] / 255f;
        return samplesF;
    }

    public AutoData(string yourPath)
    {
        this.source = yourPath;

        // hardcoded urls from my github
        string trainDataUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/train-images.idx3-ubyte";
        string trainLabelUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/train-labels.idx1-ubyte";
        string testDataUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/t10k-images.idx3-ubyte";
        string testnLabelUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/t10k-labels.idx1-ubyte";

        byte[] test, training;

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
            training = new HttpClient().GetAsync(trainDataUrl).Result.Content.ReadAsByteArrayAsync().Result.Skip(16).Take(60000 * 784).ToArray();
            this.labelsTraining = new HttpClient().GetAsync(trainLabelUrl).Result.Content.ReadAsByteArrayAsync().Result.Skip(8).Take(60000).ToArray();
            test = new HttpClient().GetAsync(testDataUrl).Result.Content.ReadAsByteArrayAsync().Result.Skip(16).Take(10000 * 784).ToArray();
            this.labelsTest = new HttpClient().GetAsync(testnLabelUrl).Result.Content.ReadAsByteArrayAsync().Result.Skip(8).Take(10000).ToArray();

            System.Console.WriteLine("Save cleaned MNIST data into folder " + yourPath + "\n");
            File.WriteAllBytes(yourPath + d1, training);
            File.WriteAllBytes(yourPath + d2, this.labelsTraining);
            File.WriteAllBytes(yourPath + d3, test);
            File.WriteAllBytes(yourPath + d4, this.labelsTest); // return;
        }
        else
        {
            // data on the system, just load from yourPath
            System.Console.WriteLine("Load MNIST data and labels from " + yourPath + "\n");
            training = File.ReadAllBytesAsync(yourPath + d1).Result.Take(60000 * 784).ToArray();
            this.labelsTraining = File.ReadAllBytesAsync(yourPath + d2).Result.Take(60000).ToArray();
            test = File.ReadAllBytesAsync(yourPath + d3).Result.Take(10000 * 784).ToArray();
            this.labelsTest = File.ReadAllBytesAsync(yourPath + d4).Result.Take(10000).ToArray();
        }

        this.samplesTrainingF = NormalizeData(labelsTraining.Length, training);
        this.samplesTestF = NormalizeData(labelsTest.Length, test);
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
