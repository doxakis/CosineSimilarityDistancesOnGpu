using Alea;
using Alea.CSharp;
using Alea.Parallel;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CosineSimilarityDistancesOnGpu
{
    public class Program
    {
        private static double[,] CosineSimilarityMultipleThread(double[][] dataset)
        {
            int size = dataset.Length * dataset.Length;
            var distances = new double[dataset.Length, dataset.Length];
            var options = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };
            Parallel.For(0, size, options, index =>
            {
                int i = index / dataset.Length;
                int j = index % dataset.Length;
                double dotProduct = 0;
                double magnitudeOne = 0;
                double magnitudeTwo = 0;
                for (int k = 0; k < dataset[i].Length; k++)
                {
                    dotProduct += (dataset[i][k] * dataset[j][k]);
                    magnitudeOne += (dataset[i][k] * dataset[i][k]);
                    magnitudeTwo += (dataset[j][k] * dataset[j][k]);
                }
                double distance = Math.Max(0, 1 - (dotProduct / Math.Sqrt(magnitudeOne * magnitudeTwo)));
                distances[i, j] = distance;
            });
            return distances;
        }

        private static double[,] CosineSimilarityGpu(Gpu gpu, double[][] dataset)
        {
            int size = dataset.Length * dataset.Length;
            var gpuDataset = gpu.Allocate(dataset);

            // Allocate directly on gpu.
            var gpuDistances = gpu.Allocate<double>(dataset.Length, dataset.Length);
            gpu.For(0, size, index =>
            {
                int i = index / dataset.Length;
                int j = index % dataset.Length;
                double dotProduct = 0;
                double magnitudeOne = 0;
                double magnitudeTwo = 0;
                for (int k = 0; k < dataset[i].Length; k++)
                {
                    dotProduct += (dataset[i][k] * dataset[j][k]);
                    magnitudeOne += (dataset[i][k] * dataset[i][k]);
                    magnitudeTwo += (dataset[j][k] * dataset[j][k]);
                }
                double distance = Math.Max(0, 1 - (dotProduct / Math.Sqrt(magnitudeOne * magnitudeTwo)));
                gpuDistances[i, j] = distance;
            });

            // Gpu -> Cpu.
            var result = new double[dataset.Length, dataset.Length];
            Gpu.Copy(gpuDistances, result);

            // Release gpu memory.
            Gpu.Free(gpuDataset);
            Gpu.Free(gpuDistances);

            return result;
        }

        static void Main(string[] args)
        {
            var gpu = Gpu.Default;

            // Clean up.
            GC.Collect();
            GC.WaitForPendingFinalizers();
            Gpu.FreeAllImplicitMemory(true);

            // Generate a random dataset.
            List<double[]> listDataset = new List<double[]>();
            int numSample = 300;
            int wordFreq = 8000;
            for (int i = 0; i < numSample; i++)
            {
                listDataset.Add(Enumerable.Range(1, wordFreq).Select(m => i + m + 1.1).ToArray());
            }
            double[][] dataset = listDataset.ToArray();

            Stopwatch watch = Stopwatch.StartNew();

            // The method can be cached so that there is only a single JIT compilation
            // It takes about 1 sec.
            CosineSimilarityGpu(gpu, new double[][] { new double[] { } });
            Console.WriteLine("JIT compilation: " + watch.Elapsed);

            // Measure Gpu.
            watch.Restart();
            var resultGpu = CosineSimilarityGpu(gpu, dataset);
            long durationGpu = watch.ElapsedMilliseconds;
            Console.WriteLine("Gpu: " + watch.Elapsed);

            // Measure Cpu.
            watch.Restart();
            var resultCpu = CosineSimilarityMultipleThread(dataset);
            long durationCpu = watch.ElapsedMilliseconds;
            Console.WriteLine("Cpu (" + Environment.ProcessorCount + " threads): " + watch.Elapsed);

            Console.WriteLine("Speed-up: " + (durationCpu / durationGpu));

            // Verify results.
            for (int i = 0; i < resultCpu.GetLength(0); i++)
            {
                for (int j = 0; j < resultCpu.GetLength(1); j++)
                {
                    double diff = Math.Abs(resultCpu[i, j] - resultGpu[i, j]);

                    // Margin of errors
                    if (diff > 0.0000001)
                    {
                        throw new Exception("Results not equals");
                    }
                }
            }

            Console.WriteLine("Press any key...");
            Console.ReadLine();
            
            var devices = Device.Devices;
            var numGpus = devices.Length;
            foreach (var device in devices)
            {
                // print device information to standard output
                device.Print();
            }

            Console.WriteLine("Press any key to continue...");
            Console.ReadLine();
        }
    }
}
