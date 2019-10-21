using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace neuralnetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            int volba=0;
            do
            {
                int inputneurons = 0; // number of input neurons         //4          //or any number
                int hiddenneurons = 0; // number of hidden neurons          //5          //or any number
                int outputneurons = 0; // number of output neurons        //3          //or any number
                int nofrows = 0;                                       //2 000  //or any number above 10
                int seed = 0;                                          //1          //or any number

                Console.WriteLine("Input the following values");
                Console.Write("N of neurons on input layer:");
                inputneurons = Convert.ToInt32(Console.ReadLine());
                Console.Write("N of neurons on output layer:");
                outputneurons = Convert.ToInt32(Console.ReadLine());
                Console.Write("N of rows:");
                nofrows = Convert.ToInt32(Console.ReadLine());
                Console.Write("N of neurons on hidden layer:");
                hiddenneurons = Convert.ToInt32(Console.ReadLine());
                Console.Write("Seed value:");
                seed = Convert.ToInt32(Console.ReadLine());

                Console.WriteLine("\nNeuralnetwork started...");

                Console.WriteLine("\nCreating " + nofrows +
                  " rows with " + inputneurons + " functions");
                double[][] allData = MakeAllData(inputneurons, hiddenneurons, outputneurons,
                  nofrows, seed);
                Console.WriteLine("Done");

                Console.WriteLine("\nPreparing training (70% of "+nofrows+" rows) and testing (30% of " + nofrows + " rows)");
                double[][] trainingdata;
                double[][] testingdata;
                devidetrainingtest(allData, 0.70, seed, out trainingdata, out testingdata);
                Console.WriteLine("Done\n");

                Console.WriteLine("Training data:");
                ZobrazMatici(trainingdata, 4, 2, true);
                Console.WriteLine("Testing data:");
                ZobrazMatici(testingdata, 4, 2, true);

                Console.WriteLine("Creating a " + inputneurons + "-" + hiddenneurons +
                  "-" + outputneurons + " neural network");
                Console.WriteLine("N of neurons on input layer:" + inputneurons + "\n" + "N of neurons on hidden layer:" + hiddenneurons + "\n" + "N of neurons on output layer:" + outputneurons + "\n" +"Seed value:"+seed+ "\n");
                NeuralNetwork nn = new NeuralNetwork(inputneurons, hiddenneurons, outputneurons);

                int maxnrows = nofrows;
                double learningspeed = 0.05;
                double velocity = 0.01;
                Console.WriteLine("\nSetting max N of rows = " + maxnrows);
                Console.WriteLine("Setting learning speed = " + learningspeed.ToString("F2"));
                Console.WriteLine("setting velocity  = " + velocity.ToString("F2"));

                Console.WriteLine("\nBeginning training");
                double[] weights = nn.traininig(trainingdata, maxnrows, learningspeed, velocity);
                Console.WriteLine("done");
                Console.WriteLine("\nweights and prejudgments konecne neuronove site:\n");
                UkazVektor(weights, 2, 10, true);

                double presnosttrenovani = nn.Accuracy(trainingdata);
                Console.WriteLine("\nKonecna presnost na trenovacich datech = " +
                  presnosttrenovani.ToString("F4"));

                double testovnipresnost = nn.Accuracy(testingdata);
                Console.WriteLine("Konecna presnost na testovnich datech = " +
                  testovnipresnost.ToString("F4"));
                Console.WriteLine("\nChcete zadat jine hodnoty? 0>ANO 1>NE");
                volba = Convert.ToInt32(Console.ReadLine());
            } while (volba==0);
            Console.WriteLine("\nKonec\n");
            Console.ReadKey();
        }

        public static void ZobrazMatici(double[][] matrix, int nofrows,
          int desetinna_mista, bool indexy)
        {
            int len = matrix.Length.ToString().Length;
            for (int i = 0; i < nofrows; ++i)
            {
                if (indexy == true)
                    Console.Write("[" + i.ToString().PadLeft(len) + "]  ");
                for (int j = 0; j < matrix[i].Length; ++j)
                {
                    double v = matrix[i][j];
                    if (v >= 0.0)
                        Console.Write(" ");
                    Console.Write(v.ToString("F" + desetinna_mista) + "  ");
                }
                Console.WriteLine("");
            }

            if (nofrows < matrix.Length)
            {
                Console.WriteLine(". . .");
                int poslednirow = matrix.Length - 1;
                if (indexy == true)
                    Console.Write("[" + poslednirow.ToString().PadLeft(len) + "]  ");
                for (int j = 0; j < matrix[poslednirow].Length; ++j)
                {
                    double v = matrix[poslednirow][j];
                    if (v >= 0.0)
                        Console.Write(" ");
                    Console.Write(v.ToString("F" + desetinna_mista) + "  ");
                }
            }
            Console.WriteLine("\n");
        }

        public static void UkazVektor(double[] vector, int desetinna_mista,
          int lineLen, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i > 0 && i % lineLen == 0) Console.WriteLine("");
                if (vector[i] >= 0) Console.Write(" ");
                Console.Write(vector[i].ToString("F" + desetinna_mista) + " ");
            }
            if (newLine == true)
                Console.WriteLine("");
        }

        static double[][] MakeAllData(int inputneurons, int hiddenneurons,
          int outputneurons, int nofrows, int seed)
        {
            Random rnd = new Random(seed);
            int numweights = (inputneurons * hiddenneurons) + hiddenneurons +
              (hiddenneurons * outputneurons) + outputneurons;
            double[] weights = new double[numweights]; 
            for (int i = 0; i < numweights; ++i)
                weights[i] = 20.0 * rnd.NextDouble() - 10.0; 

            Console.WriteLine("Creating weights and prejudgments:");
            UkazVektor(weights, 2, 10, true);

            double[][] result = new double[nofrows][]; 
            for (int i = 0; i < nofrows; ++i)
                result[i] = new double[inputneurons + outputneurons]; 

            NeuralNetwork gnn =
              new NeuralNetwork(inputneurons, hiddenneurons, outputneurons);
            gnn.setweights(weights);

            for (int r = 0; r < nofrows; ++r)
            {
                double[] inputs = new double[inputneurons];
                for (int i = 0; i < inputneurons; ++i)
                    inputs[i] = 20.0 * rnd.NextDouble() - 10.0; 
                double[] outputs = gnn.Computeoutputs(inputs);

                double[] jeden_z_n = new double[outputneurons];

                int maxIndex = 0;
                double maxValue = outputs[0];
                for (int i = 0; i < outputneurons; ++i)
                {
                    if (outputs[i] > maxValue)
                    {
                        maxIndex = i;
                        maxValue = outputs[i];
                    }
                }
                jeden_z_n[maxIndex] = 1.0;

                
                int c = 0; 
                for (int i = 0; i < inputneurons; ++i) 
                    result[r][c++] = inputs[i];
                for (int i = 0; i < outputneurons; ++i) 
                    result[r][c++] = jeden_z_n[i];
            } 
            return result;
        } /

        static void devidetrainingtest(double[][] allData, double trainPct,
          int seed, out double[][] trainingdata, out double[][] testingdata)
        {
            Random rnd = new Random(seed);
            int totRows = allData.Length;
            int numTrainRows = (int)(totRows * trainPct); 
            int numTestRows = totRows - numTrainRows;
            trainingdata = new double[numTrainRows][];
            testingdata = new double[numTestRows][];

            double[][] copy = new double[allData.Length][]; 
            for (int i = 0; i < copy.Length; ++i)
                copy[i] = allData[i];

            for (int i = 0; i < copy.Length; ++i) 
            {
                int r = rnd.Next(i, copy.Length);
                double[] tmp = copy[r];
                copy[r] = copy[i];
                copy[i] = tmp;
            }
            for (int i = 0; i < numTrainRows; ++i)
                trainingdata[i] = copy[i];

            for (int i = 0; i < numTestRows; ++i)
                testingdata[i] = copy[i + numTrainRows];
        } 

    }

    public class NeuralNetwork
    {
        private int inputneurons; 
        private int hiddenneurons; 
        private int outputneurons; 

        private double[] inputs;
        private double[][] ihweights; 
        private double[] hBiases;
        private double[] houtputs;

        private double[][] howeights; 
        private double[] oBiases;
        private double[] outputs;

        private Random rnd;

        public NeuralNetwork(int inputneurons, int hiddenneurons, int outputneurons)
        {
            this.inputneurons = inputneurons;
            this.hiddenneurons = hiddenneurons;
            this.outputneurons = outputneurons;

            this.inputs = new double[inputneurons];

            this.ihweights = ceratematrix(inputneurons, hiddenneurons, 0.0);
            this.hBiases = new double[hiddenneurons];
            this.houtputs = new double[hiddenneurons];

            this.howeights = ceratematrix(hiddenneurons, outputneurons, 0.0);
            this.oBiases = new double[outputneurons];
            this.outputs = new double[outputneurons];

            this.rnd = new Random(0);
            this.runweights(); 
        } 

        private static double[][] ceratematrix(int rows,
          int cols, double v) 
        {
            double[][] result = new double[rows][];
            for (int r = 0; r < result.Length; ++r)
                result[r] = new double[cols];
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result[i][j] = v;
            return result;
        }

        private void runweights() 
        {
            int numweights = (inputneurons * hiddenneurons) +
              (hiddenneurons * outputneurons) + hiddenneurons + outputneurons;
            double[] initialweights = new double[numweights];
            for (int i = 0; i < initialweights.Length; ++i)
                initialweights[i] = (0.001 - 0.0001) * rnd.NextDouble() + 0.0001;
            this.setweights(initialweights);
        }

        public void setweights(double[] weights)
        {
            int numweights = (inputneurons * hiddenneurons) +
              (hiddenneurons * outputneurons) + hiddenneurons + outputneurons;
            if (weights.Length != numweights)
                throw new Exception("Invalid weights array in setweights");

            int k = 0; 

            for (int i = 0; i < inputneurons; ++i)
                for (int j = 0; j < hiddenneurons; ++j)
                    ihweights[i][j] = weights[k++];
            for (int i = 0; i < hiddenneurons; ++i)
                hBiases[i] = weights[k++];
            for (int i = 0; i < hiddenneurons; ++i)
                for (int j = 0; j < outputneurons; ++j)
                    howeights[i][j] = weights[k++];
            for (int i = 0; i < outputneurons; ++i)
                oBiases[i] = weights[k++];
        }

        public double[] Getweights()
        {
            int numweights = (inputneurons * hiddenneurons) +
              (hiddenneurons * outputneurons) + hiddenneurons + outputneurons;
            double[] result = new double[numweights];
            int k = 0;
            for (int i = 0; i < ihweights.Length; ++i)
                for (int j = 0; j < ihweights[0].Length; ++j)
                    result[k++] = ihweights[i][j];
            for (int i = 0; i < hBiases.Length; ++i)
                result[k++] = hBiases[i];
            for (int i = 0; i < howeights.Length; ++i)
                for (int j = 0; j < howeights[0].Length; ++j)
                    result[k++] = howeights[i][j];
            for (int i = 0; i < oBiases.Length; ++i)
                result[k++] = oBiases[i];
            return result;
        }

        public double[] Computeoutputs(double[] xvalues)
        {
            double[] hSums = new double[hiddenneurons]; 
            double[] oSums = new double[outputneurons]; 
            for (int i = 0; i < xvalues.Length; ++i) 
                this.inputs[i] = xvalues[i];

            for (int j = 0; j < hiddenneurons; ++j)  
                for (int i = 0; i < inputneurons; ++i)
                    hSums[j] += this.inputs[i] * this.ihweights[i][j]; 

            for (int i = 0; i < hiddenneurons; ++i)  
                hSums[i] += this.hBiases[i];

            for (int i = 0; i < hiddenneurons; ++i)   
                this.houtputs[i] = HyperTan(hSums[i]); 
            for (int j = 0; j < outputneurons; ++j)   
                for (int i = 0; i < hiddenneurons; ++i)
                    oSums[j] += houtputs[i] * howeights[i][j];

            for (int i = 0; i < outputneurons; ++i)  
                oSums[i] += oBiases[i];

            double[] softOut = Softmax(oSums); 
            Array.Copy(softOut, outputs, softOut.Length);

            double[] retresult = new double[outputneurons]; 
            Array.Copy(this.outputs, retresult, retresult.Length);
            return retresult;
        }

        private static double HyperTan(double x)
        {
            if (x < -20.0) return -1.0; 
            else if (x > 20.0) return 1.0;
            else return Math.Tanh(x);
        }

        private static double[] Softmax(double[] oSums)
        {
            double sum = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
                sum += Math.Exp(oSums[i]);

            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
                result[i] = Math.Exp(oSums[i]) / sum;

            return result; 
        }

        public double[] traininig(double[][] trainingdata, int maxnrows,
          double learningspeed, double velocity)
        {
            double[][] hoGrads = ceratematrix(hiddenneurons, outputneurons, 0.0);
            double[] obGrads = new double[outputneurons];            

            double[][] ihGrads = ceratematrix(inputneurons, hiddenneurons,
            double[] hbGrads = new double[hiddenneurons];
            double[] oSignals = new double[outputneurons];                  
            double[] hSignals = new double[hiddenneurons];      
            double[][] ihPrevweightsDelta = ceratematrix(inputneurons, hiddenneurons, 0.0);
            double[] hPrevBiasesDelta = new double[hiddenneurons];
            double[][] hoPrevweightsDelta = ceratematrix(hiddenneurons, outputneurons, 0.0);
            double[] oPrevBiasesDelta = new double[outputneurons];

            int row = 0;
            double[] xvalues = new double[inputneurons]; 
            double[] tValues = new double[outputneurons]; 
            double derivative = 0.0;
            double errorSignal = 0.0;

            int[] sequence = new int[trainingdata.Length];
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            int errInterval = maxnrows / 15; 
            while (row < maxnrows)
            {
                ++row;

                if (row % errInterval == 0 && row <= maxnrows)
                {
                    double trainErr = Error(trainingdata);
                    double trainErr1 = Error(trainingdata)*100.00;
                    Console.WriteLine("row = " + row + "  error = " +
                      trainErr.ToString("F4") +" in percentage > "+ trainErr1.ToString("F4")+"%");
                }

                Shuffle(sequence); 
                for (int ii = 0; ii < trainingdata.Length; ++ii)
                {
                    int idx = sequence[ii];
                    Array.Copy(trainingdata[idx], xvalues, inputneurons);
                    Array.Copy(trainingdata[idx], inputneurons, tValues, 0, outputneurons);
                    Computeoutputs(xvalues); 

                    for (int k = 0; k < outputneurons; ++k)
                    {
                        errorSignal = tValues[k] - outputs[k];  
                        derivative = (1 - outputs[k]) * outputs[k]; 
                        oSignals[k] = errorSignal * derivative;
                    }

                    for (int j = 0; j < hiddenneurons; ++j)
                        for (int k = 0; k < outputneurons; ++k)
                            hoGrads[j][k] = oSignals[k] * houtputs[j];

                    for (int k = 0; k < outputneurons; ++k)
                        obGrads[k] = oSignals[k] * 1.0; 

                    for (int j = 0; j < hiddenneurons; ++j)
                    {
                        derivative = (1 + houtputs[j]) * (1 - houtputs[j]);
                        double sum = 0.0; 
                        for (int k = 0; k < outputneurons; ++k)
                        {
                            sum += oSignals[k] * howeights[j][k]; 
                        }
                        hSignals[j] = derivative * sum;
                    }

                    for (int i = 0; i < inputneurons; ++i)
                        for (int j = 0; j < hiddenneurons; ++j)
                            ihGrads[i][j] = hSignals[j] * inputs[i];

                    for (int j = 0; j < hiddenneurons; ++j)
                        hbGrads[j] = hSignals[j] * 1.0; 

                    for (int i = 0; i < inputneurons; ++i)
                    {
                        for (int j = 0; j < hiddenneurons; ++j)
                        {
                            double delta = ihGrads[i][j] * learningspeed;
                            ihweights[i][j] += delta; 
                            ihweights[i][j] += ihPrevweightsDelta[i][j] * velocity;
                            ihPrevweightsDelta[i][j] = delta; 
                        }
                    }

                    for (int j = 0; j < hiddenneurons; ++j)
                    {
                        double delta = hbGrads[j] * learningspeed;
                        hBiases[j] += delta;
                        hBiases[j] += hPrevBiasesDelta[j] * velocity;
                        hPrevBiasesDelta[j] = delta;
                    }

                    for (int j = 0; j < hiddenneurons; ++j)
                    {
                        for (int k = 0; k < outputneurons; ++k)
                        {
                            double delta = hoGrads[j][k] * learningspeed;
                            howeights[j][k] += delta;
                            howeights[j][k] += hoPrevweightsDelta[j][k] * velocity;
                            hoPrevweightsDelta[j][k] = delta;
                        }
                    }

                    for (int k = 0; k < outputneurons; ++k)
                    {
                        double delta = obGrads[k] * learningspeed;
                        oBiases[k] += delta;
                        oBiases[k] += oPrevBiasesDelta[k] * velocity;
                        oPrevBiasesDelta[k] = delta;
                    }

                } 

            } 
            double[] bestWts = Getweights();
            return bestWts;
        } 

        private void Shuffle(int[] sequence)
        {
            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = this.rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        } 

        private double Error(double[][] trainingdata)
        {
            double sumSquaredError = 0.0;
            double[] xvalues = new double[inputneurons]; 
            double[] tValues = new double[outputneurons]; 
            for (int i = 0; i < trainingdata.Length; ++i)
            {
                Array.Copy(trainingdata[i], xvalues, inputneurons);
                Array.Copy(trainingdata[i], inputneurons, tValues, 0, outputneurons); 
                double[] yValues = this.Computeoutputs(xvalues); 
                for (int j = 0; j < outputneurons; ++j)
                {
                    double err = tValues[j] - yValues[j];
                    sumSquaredError += err * err;
                }
            }
            return sumSquaredError / trainingdata.Length;
        }

        public double Accuracy(double[][] testingdata)
        {
            int numCorrect = 0;
            int numWrong = 0;
            double[] xvalues = new double[inputneurons];
            double[] tValues = new double[outputneurons];
            double[] yValues;

            for (int i = 0; i < testingdata.Length; ++i)
            {
                Array.Copy(testingdata[i], xvalues, inputneurons); 
                Array.Copy(testingdata[i], inputneurons, tValues, 0, outputneurons); 
                yValues = this.Computeoutputs(xvalues);
                int maxIndex = MaxIndex(yValues); 
                int tMaxIndex = MaxIndex(tValues);

                if (maxIndex == tMaxIndex)
                    ++numCorrect;
                else
                    ++numWrong;
            }
            return (numCorrect * 1.0) / (numCorrect + numWrong);
        }

        private static int MaxIndex(double[] vector) 
        {
            int bigIndex = 0;
            double biggestVal = vector[0];
            for (int i = 0; i < vector.Length; ++i)
            {
                if (vector[i] > biggestVal)
                {
                    biggestVal = vector[i];
                    bigIndex = i;
                }
            }
            return bigIndex;
        }
    }
}
