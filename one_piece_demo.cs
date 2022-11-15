// for more info https://github.com/grensen/good_vs_bad_code
System.Action<string> print = System.Console.WriteLine;

#if DEBUG
System.Console.WriteLine("Debug mode is on, switch to Release mode");
#endif 

print("\nBegin one piece demo .NET 7\n");

// get data
AutoData d = new(@"C:\mnist\");

// define neural network 
int[] network = { 784, 100, 100, 10 };
var LEARNINGRATE = 0.0005f;
var MOMENTUM = 0.67f;
var EPOCHS = 10;
var BATCHSIZE = 800;
var FACTOR = 0.99f;

NewRunDemo(network, d, LEARNINGRATE, MOMENTUM, FACTOR, EPOCHS, BATCHSIZE);

print("\nEnd good code vs. bad code demo");

//+---------------------------------------------------------------------+
static void NewRunDemo(int[] network, AutoData d, float LEARNINGRATE, float MOMENTUM, float FACTOR, int EPOCHS, int BATCHSIZE)
{
    System.Console.WriteLine("NETWORK      = " + string.Join("-", network));
    System.Console.WriteLine("LEARNINGRATE = " + LEARNINGRATE);
    System.Console.WriteLine("MOMENTUM     = " + MOMENTUM);
    System.Console.WriteLine("BATCHSIZE    = " + BATCHSIZE);
    System.Console.WriteLine("EPOCHS       = " + EPOCHS + " (50 * 60,000 = 3,000,000 exampels)");
    System.Console.WriteLine("FACTOR       = " + FACTOR + "");

    Net nn = new(network);

    DateTime elapsed = DateTime.Now;
    System.Console.WriteLine("\nStart training");
    RunTraining(true, false, elapsed, d, nn, 60000, LEARNINGRATE, MOMENTUM, FACTOR, EPOCHS, BATCHSIZE);
    System.Console.WriteLine("\nStart testing");
    RunTraining(false, false, elapsed, d, nn, 10000);
    var time = (float)((DateTime.Now - elapsed).TotalMilliseconds / 1000.0f);

    System.Console.WriteLine("\nTotal time good code = " + (time).ToString("F2") + "s"
        + ", reference code = " + (56.13).ToString("F2") + "s");
    System.Console.WriteLine("\nGood code Single time = " + time.ToString("F2") + "s");

    static void RunTraining(bool training, bool multiCore, DateTime elapsed, AutoData d, Net neural, int len, float lr = 0, float mom = 0, float FACTOR = 1, int EPOCHS = 1, int BATCHSIZE = 0)
    {
        float[] delta = new float[neural.weights.Length];
        byte[] samples = training ? d.samplesTraining : d.samplesTest;
        byte[] lables = training ? d.labelsTraining : d.labelsTest;
        for (int epoch = 0, B = training ? len / BATCHSIZE : 1; epoch < EPOCHS; epoch++, lr *= FACTOR, mom *= FACTOR)
        {
            bool[] c = new bool[training ? B * BATCHSIZE : len];
            for (int b = 0; b < B; b++)
            {
                if (multiCore)
                    System.Threading.Tasks.Parallel.ForEach(
                        System.Collections.Concurrent.Partitioner.Create(b * BATCHSIZE, training ? (b + 1) * BATCHSIZE : len), range =>
                        {
                            for (int x = range.Item1, X = range.Item2; x < X; x++)
                                c[x] = EvalOrTrain(training, x, samples, neural.neuronLen, neural.net, neural.weights, delta, lables[x]);
                        });             
                else                                    
                    for (int x = b * BATCHSIZE, X = training ? (b + 1) * BATCHSIZE : len; x < X; x++)
                        c[x] = EvalOrTrain(training, x, samples, neural.neuronLen, neural.net, neural.weights, delta, lables[x]);
                 
                if (training)
                    System.Threading.Tasks.Parallel.ForEach(
                        System.Collections.Concurrent.Partitioner.Create(0, neural.weights.Length), range =>
                        {
                            UpdateWeights(range.Item1, range.Item2, neural.weights, delta, lr, mom);
                        });                
            }
            int correct = c.Count(n => n);
           // if (!training || (epoch + 1) % 10 == 0) 
                PrintInfo("Epoch = " + (1 + epoch), correct, training ? B * BATCHSIZE : len, elapsed);
        }
    }  
    static bool EvalOrTrain(bool training, int x, byte[] s, int neuronLen, int[] net, float[] wts, float[] d, byte t)
    {
        // 0 get space on the machine
        int L = net.Length - 1, N = neuronLen, IH = N - net[^1], p = net[^1] - 1;
        Span<float> n = stackalloc float[N], g = stackalloc float[N];

        // 1 feed input and normalize
        for (int i = 0, ii = x * 784; i < 784; i++)
            n[i] = s[ii + i] > 0 ? s[ii + i] / 255f : 0;

        // 2 feed forward
        for (int k = net[0], w = 0, i = 0; i < L; i++, k += net[i])
            for (int l = k - net[i]; l < k; l++, w += net[i + 1])
                if (n[l] > 0)
                    for (int r = 0; r < net[i + 1]; r++)
                        n[r + k] += wts[r + w] * n[l];

        // 3 or 4 argmax // return here if testing  
        float max = n[N - 1], scale = 0;
        for (int i = p - 1; i >= 0; i--)
            max = n[i + IH] > max ? n[(p = i) + IH] : max;

        // 4 or 3 softmax
        for (int r = IH; r < N; r++) scale += n[r] = MathF.Exp(n[r]);
        for (int l = N - 1; l >= IH; l--) n[l] /= scale;

        // faster training because we dont need to train if prop == 1, 0.99 is even faster 
        if (!training || n[IH + t] >= 0.99) return p == t;

        // 5 backprop     
        for (int r = IH, i = 0; r < N; r++, i++)
            g[r] = t == i ? 1 - n[r] : -n[r];

        for (int i = L - 1, w = wts.Length - 1; i >= 0; i--, IH -= net[i + 1])
            for (int l = IH - 1; l >= IH - net[i]; l--)
                if (n[l] > 0)
                    for (int r = IH + net[i + 1] - 1; r >= IH; r--, w--)
                    {
                        g[l] += wts[w] * g[r];
                        d[w] += n[l] * g[r];
                    }
                else w -= net[i + 1];

        // 6 prediction
        return p == t;
    }
    static void UpdateWeights(int st, int en, float[] weights, float[] delta, float lr, float mom)
    {
        for (int w = st; w < en; w++)
        {
            weights[w] += delta[w] * lr;
            delta[w] *= mom;
        }
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

            for (int i = 0, w = 0; i < net.Length - 1; i++, w += net[i] * net[i - 1]) // layer
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