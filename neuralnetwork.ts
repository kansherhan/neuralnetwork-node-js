class Utils
{
    static sigmoid(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }

    static dsigmoid(x: number): number {
		return x * (1 - x);
	}

    static argmax(arr: Array<any>): number {
        let maxValueIndex = 0;
        let maxValue = 0;

        for (let i = 0; i < arr.length; i++) {
			if (arr[i] > maxValue)
			{
				maxValue = arr[i];
				maxValueIndex = i;
			}
		}

        return maxValueIndex;
    }
}

class Topology
{
    readonly learningRate: number;
    readonly layerCount: number;

    readonly layerSize: Array<number>;

    constructor(learningRate: number, size: Array<number>) {
        this.learningRate = learningRate;
        this.layerCount = size.length;
        this.layerSize = size;
    }
}

class Layer
{
    readonly size: number;
    readonly nextLayerSize: number;

    neurons: Array<number>;
    biases: Array<number>;

    weights: Array<Array<number>>;
    
    static getValueDefault = (): number => Math.random() * 2.0 - 1.0;

    constructor(size: number, nextLayerSize: number) {
        this.size = size;
        this.nextLayerSize = nextLayerSize;

        this.neurons = new Array<number>(size);
        this.biases = new Array<number>(size);

        this.weights = new Array<Array<number>>(size);

        for (let i = 0; i < size; i++) {
            this.neurons[i] = Layer.getValueDefault();
            this.biases[i] = Layer.getValueDefault();

            this.weights[i] = new Array();

            for (let j = 0; j < nextLayerSize; j++) {
                this.weights[i][j] = Layer.getValueDefault();
            }
        }
    }
}

class Network
{
    readonly topology: Topology;
    readonly layers: Array<Layer>;

    iterationCount: number;

    constructor(topology: Topology) {
        this.topology = topology;
        
        this.layers = new Array<Layer>(topology.layerCount);

        this.iterationCount = 0;

        let layerSize = topology.layerSize;

        for (let i = 0; i < topology.layerCount; i++) {
            let nextLayerSize = 0;

            if (i < topology.layerCount - 1)
            {
                nextLayerSize = layerSize[i + 1];
            }

            this.layers[i] = new Layer(layerSize[i], nextLayerSize);
        }
    }

    feedForword(inputs: Array<number>): Array<number> {
        this.layers[0].neurons = inputs;

        for (let i = 1; i < this.topology.layerCount; i++) {
            const currentLayer: Layer = this.layers[i - 1];
            const nextLayer: Layer = this.layers[i];

            for (let j = 0; j < nextLayer.size; j++) {
                nextLayer.neurons[j] = 0.0;

                for (let k = 0; k < currentLayer.size; k++) {
                    nextLayer.neurons[j] += currentLayer.neurons[k] * currentLayer.weights[k][j];
                }

                nextLayer.neurons[j] += nextLayer.biases[j];
                nextLayer.neurons[j] = Utils.sigmoid(nextLayer.neurons[j]);
            }
        }

        return this.layers[this.topology.layerCount - 1].neurons;
    }

    backpropagation(targets: Array<number>, inputs: Array<number>): Array<number> {
        let networkOutputs = this.feedForword(inputs);
        this.iterationCount += 1;

        let lastLayer: Layer = this.layers[this.topology.layerCount - 1];
        let errors = new Array<number>(lastLayer.size);

        for (let i = 0; i < lastLayer.size; i++) {
			errors[i] = targets[i] - lastLayer.neurons[i];
		}

        for (let k = this.layers.length - 2; k >= 0; k--) {
			let currentLayer: Layer = this.layers[k];
			let nextLayer: Layer = this.layers[k + 1];

			let gradients = new Array<number>(nextLayer.size);
			for (let i = 0; i < gradients.length; i++) {
				gradients[i] = errors[i] * Utils.dsigmoid(nextLayer.neurons[i]) * this.topology.learningRate;
			}

			let deltas = new Array<Array<number>>(nextLayer.size);
			for (let i = 0; i < nextLayer.size; i++) {
				deltas[i] = new Array(currentLayer.size);

				for (let j = 0; j < currentLayer.size; j++) {
					deltas[i][j] = gradients[i] * currentLayer.neurons[j];
				}
			}

			let newErrors = new Array<number>(currentLayer.size);
			for (let i = 0; i < currentLayer.size; i++) {
				newErrors[i] = 0.0;

				for (let j = 0; j < nextLayer.size; j++) {
					newErrors[i] += currentLayer.weights[i][j] * errors[j];
				}
			}

			errors = Object.freeze(newErrors) as Array<number>;

			for (let i = 0; i < nextLayer.size; i++) {
				for (let j = 0; j < currentLayer.size; j++) {
					currentLayer.weights[j][i] += deltas[i][j];
				}
			}

			for (let i = 0; i < nextLayer.size; i++) {
				nextLayer.biases[i] += gradients[i];
			}
		}

        return networkOutputs;
    }
}

export { Utils, Layer, Topology, Network };
