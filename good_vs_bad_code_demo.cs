// for more info https://github.com/grensen/good_vs_bad_code
// based on https://github.com/grensen/multi-core
#if DEBUG
System.Console.WriteLine("Debug mode is on, switch to Release mode");
#endif 
System.Action<string> print = System.Console.WriteLine;

print("\nBegin good code vs. bad code demo\n");

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

print("\nEnd good code vs. bad code demo");

//+---------------------------------------------------------------------+

static void RunDemo(int[] network, AutoData d, float LEARNINGRATE, float MOMENTUM, float FACTOR, int EPOCHS, int BATCHSIZE)
{
    System.Console.WriteLine("NETWORK      = " + string.Join("-", network));
    System.Console.WriteLine("LEARNINGRATE = " + LEARNINGRATE);
    System.Console.WriteLine("MOMENTUM     = " + MOMENTUM);
    System.Console.WriteLine("BATCHSIZE    = " + BATCHSIZE);
    System.Console.WriteLine("EPOCHS       = " + EPOCHS + " (50 * 60,000 = 3,000,000 exampels)");
    System.Console.WriteLine("FACTOR       = " + FACTOR + "");

    var sBadTime = 0.0f;
    if (!false)
    {
        Net badSingleNet = new(network);
        System.Console.WriteLine("\nStart bad code reproducible Single-Core training");
        sBadTime = BadRunNet(false, d, badSingleNet, 60000, LEARNINGRATE, MOMENTUM, FACTOR, EPOCHS, BATCHSIZE);
    }
    var mBadTime = 0.0f;
    if (!false)
    {
        Net badMultiNet = new(network);
        System.Console.WriteLine("\nStart bad code non reproducible Multi-Core training");
        mBadTime = BadRunNet(true, d, badMultiNet, 60000, LEARNINGRATE, MOMENTUM, FACTOR, EPOCHS, BATCHSIZE);
    }
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
        + ", reference code = " + (56.13).ToString("F2") + "s"
        + ", bad code = " + (sBadTime + mBadTime).ToString("F2") + "s");

    System.Console.WriteLine("\nGood code Single time = " + sGoodTime.ToString("F2") + "s" + " good code Multi time = " + mGoodTime.ToString("F2") + "s");
    System.Console.WriteLine("Bad code Single time = " + sBadTime.ToString("F2") + "s" + " bad code Multi time = " + mBadTime.ToString("F2") + "s");

    System.Console.WriteLine("\nGood code Multi-Core was " + (sGoodTime / mGoodTime).ToString("F2") + " times faster than good code Single-Core");
    System.Console.WriteLine("Bad code Multi-Core was " + (sBadTime / mBadTime).ToString("F2") + " times faster than bad code Single-Core");

    System.Console.WriteLine("\nGood code was " + (56.13 / (sGoodTime + mGoodTime)).ToString("F2") + " times faster than reference code");
    System.Console.WriteLine("Reference code was " + ((sBadTime + mBadTime) / 56.13).ToString("F2") + " times faster than bad code");
    System.Console.WriteLine("Good code was " + ((sBadTime + mBadTime) / (sGoodTime + mGoodTime)).ToString("F2") + " times faster than bad code");
    System.Console.WriteLine("Good code Multi was " + (sBadTime / mGoodTime).ToString("F2") + " times faster than bad code Single");
}

static float BadRunNet(bool multiCore, AutoData d, Net neural, int len, float lr, float mom, float FACTOR, int EPOCHS, int BATCHSIZE)
{
    DateTime elapsed = DateTime.Now;
    BadRunTraining(multiCore, elapsed, d, neural, len, lr, mom, FACTOR, EPOCHS, BATCHSIZE);
    return BadRunTest(multiCore, elapsed, d, neural, 10000);

    static void BadRunTraining(bool multiCore, DateTime elapsed, AutoData d, Net neural, int len, float lr, float mom, float FACTOR, int EPOCHS, int BATCHSIZE)
    {
        int B = len / BATCHSIZE;
        float[] delta = new float[neural.weights.Length];

        for (int epoch = 0; epoch < EPOCHS; epoch++, lr *= FACTOR, mom *= FACTOR)
        {

            int correct = 0;
            for (int b = 0; b < B; b++)
            {
                bool[] c = new bool[BATCHSIZE];
                int start = b * BATCHSIZE, end = (b + 1) * BATCHSIZE;
                if (multiCore)
                    System.Threading.Tasks.Parallel.ForEach(
                        System.Collections.Concurrent.Partitioner.Create(start, end), range =>
                        {
                            for (int x = range.Item1, X = range.Item2; x < X; x++)
                                c[x - start] = BadEvalAndTrain(x, d.labelsTraining[x], d.samplesTraining, neural, delta);
                        });
                else
                    for (int x = start, X = end; x < X; x++)
                        c[x - start] = BadEvalAndTrain(x, d.labelsTraining[x], d.samplesTraining, neural, delta);

                for (int i = 0; i < BATCHSIZE; i++) if (c[i]) correct++;

                BadUpdate(neural.weights, delta, lr, mom);
            }
            //int correct = c.Count(n => n); // for (int i = 0; i < len; i++) if (c[i]) correct++;
            if ((epoch + 1) % 10 == 0)
                PrintInfo("Epoch = " + (1 + epoch), correct, B * BATCHSIZE, elapsed);
        }
    }
    static float BadRunTest(bool multiCore, DateTime elapsed, AutoData d, Net neural, int len)
    {
        bool[] c = new bool[len];
        if (multiCore)
            System.Threading.Tasks.Parallel.ForEach(System.Collections.Concurrent.Partitioner.Create(0, len), range =>
            {
                for (int x = range.Item1; x < range.Item2; x++)
                    c[x] = BadEvalTest(x, d.labelsTest[x], d.samplesTest, neural);
            });
        else
            for (int x = 0; x < len; x++)
                c[x] = BadEvalTest(x, d.labelsTest[x], d.samplesTest, neural);

        int correct = c.Count(n => n); // int correct = 0; //for (int i = 0; i < c.Length; i++) if (c[i]) correct++;
        PrintInfo("Test", correct, 10000, elapsed);
        return (float)((DateTime.Now - elapsed).TotalMilliseconds / 1000.0f);
    }
}
static bool BadEvalAndTrain(int x, byte t, byte[] samples, Net neural, float[] delta)
{
    Span<float> neuron = new float[neural.neuronLen];

    int p = BadEval(x, samples, neural, neuron);
    if (neuron[neural.neuronLen - neural.net[^1] + t] < 0.99)
        BadBP(neural, neuron, delta, t);
    return p == t;

    static void BadBP(Net neural, Span<float> neuron, float[] delta, int target)
    {
        int nLen = neuron.Length;
        int inputHidden = nLen - neural.net[neural.layers]; // size of input and hidden neurons
        Span<float> gradient = new float[nLen];

        // output error gradients, hard target as 1 for its class
        for (int r = inputHidden, p = 0; r < nLen; r++, p++)
            gradient[r] = target == p ? 1 - neuron[r] : -neuron[r];

        for (int i = neural.layers - 1, j = inputHidden, k = nLen, m = neural.weights.Length; i >= 0; i--)
        {
            int right = neural.net[i + 1], left = neural.net[i];
            j -= left; m -= right * left; k -= right;

            for (int l = j, L = left + j, w = m; l < L; l++, w += right)
            {
                float n = neuron[l];
                if (n > 0)
                    for (int r = 0; r < right; r++)
                        gradient[l] = gradient[l] + neural.weights[w + r] * gradient[k + r];

                for (int r = 0; r < right; r++)
                    delta[w + r] = delta[w + r] + neuron[l] * gradient[k + r];
            }
        }
    }
}
static bool BadEvalTest(int x, byte t, byte[] samples, Net neural)
{
    Span<float> neuron = new float[neural.neuronLen];
    int p = BadEval(x, samples, neural, neuron);
    return p == t;
}
static int BadEval(int x, byte[] samples, Net neural, Span<float> neuron)
{
    BadFeedInput(x, samples, neuron);
    BadFeedForward(neuron, neural);
    BadSoftmax(neuron, neural.net[neural.layers]);
    return BadArgmax(neural, neuron);

    static void BadFeedInput(int x, byte[] samples, Span<float> neuron)
    {
        for (int i = 0, ii = x * 784; i < 784; i++)
            if (samples[ii + i] > 0)
                neuron[i] = samples[ii + i] / 255f;
    }
    static void BadFeedForward(Span<float> neuron, Net neural)
    {
        for (int k = neural.net[0], w = 0, i = 0; i < neural.layers; i++)
        {
            int right = neural.net[i + 1];
            for (int l = k - neural.net[i]; l < k; l++)
                for (int r = 0; r < right; r++, w++)
                    neuron[k + r] += neuron[l] * neural.weights[w];

            if (i != neural.layers - 1)
                for (int r = 0; r < right; r++) // ReLU activation
                    neuron[k + r] = neuron[k + r] > 0 ? neuron[k + r] : 0;
            k += right;
        }
    }

    static void BadSoftmax(Span<float> neuron, int output)
    {
        int N = neuron.Length;
        float scale = 0;
        for (int n = N - output; n < N; n++)
            neuron[n] = MathF.Exp(neuron[n]);
        for (int n = N - output; n < N; n++)
            scale += neuron[n];
        for (int n = N - output; n < N; n++)
            neuron[n] /= scale;
    }
    static int BadArgmax(Net neural, Span<float> neuron)
    {
        int N = neuron.Length;
        int output = neural.net[neural.layers];
        int prediction = 0;
        float max = float.MinValue;
        for (int i = 0; i < output; i++)
        {
            float n = neuron[i + N - output];
            if (n > max) { max = n; prediction = i; } // grab maxout prediction here
        }
        return prediction;
    }
}
static void BadUpdate(float[] weight, float[] delta, float lr, float mom)
{
    for (int w = 0; w < weight.Length; w++)
        weight[w] = weight[w] + delta[w] * lr;
    for (int w = 0; w < weight.Length; w++)
        delta[w] = delta[w] * mom;
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
    Span<float> neuron = stackalloc float[neural.neuronLen];
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
    Span<float> neuron = stackalloc float[neural.neuronLen];
    int p = Eval(x, neural, samples, neuron);
    return p == t;
}
static int Eval(int x, Net neural, byte[] samples, Span<float> neuron)
{
    FeedInput(x, samples, neuron);
    FeedForward(neuron, neural);
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
    static void FeedForward(Span<float> neuron, Net neural)
    {
        for (int k = neural.net[0], w = 0, i = 0; i < neural.layers; i++)
        {
            int right = neural.net[i + 1];
            for (int l = k - neural.net[i]; l < k; l++, w += right)
            {
                float n = neuron[l];
                if (n <= 0) continue;
                int r = 0;
                for (; r < right - 8; r += 8)
                {
                    int wr = w + r, kr = k + r;
                    float p = neural.weights[wr] * n; neuron[kr] += p;
                    p = neural.weights[wr + 1] * n; neuron[kr + 1] += p;
                    p = neural.weights[wr + 2] * n; neuron[kr + 2] += p;
                    p = neural.weights[wr + 3] * n; neuron[kr + 3] += p;
                    p = neural.weights[wr + 4] * n; neuron[kr + 4] += p;
                    p = neural.weights[wr + 5] * n; neuron[kr + 5] += p;
                    p = neural.weights[wr + 6] * n; neuron[kr + 6] += p;
                    p = neural.weights[wr + 7] * n; neuron[kr + 7] += p;
                }
                // source loop for (; r < right; r++) { float p = neural.weights[r + w] * n; neuron[r + k] += p; }
                for (; r < right; r++) { float p = neural.weights[r + w] * n; neuron[r + k] += p; }
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
