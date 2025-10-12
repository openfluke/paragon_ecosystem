using OpenFluke;
using OpenFluke.Bench;

class Program
{
    static void Main()
    {
        var bench = new BenchSuite(Portal.Init);
        bench.EnableCsv("bench.csv");
        bench.RunAll(Presets.MNIST_ZOO);
    }
}
