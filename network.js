class Utils
{
	static sigmoid(x) {
		return 1 / (1 + Math.exp(-x));
	}
	
	static dsigmoid(x) {
		return x * (1 - x);
	}

	static createArray(length, methodValue) {
		let arr = new Array(length);

		for (let i = 0; i < arr.length; i++) {
			arr[i] = methodValue();
		}

		return arr;
	}

	static createTwoWorldArray(length, longLength, methodValue) {
		let arr = new Array(length);
	
		for (let i = 0; i < length; i++) {
			arr[i] = new Array(longLength);

			for (let j = 0; j < arr[i].length; j++) {
				arr[i][j] = methodValue();
			}
		}
	
		return arr;
	}

	static argmax(arr) {
		let max_value_index = 0;
		let max_value = 0;

		for (let i = 0; i < arr.length; i++) {
			if (arr[i] > max_value)
			{
				max_value = arr[i];
				max_value_index = i;
			}
		}

		return max_value_index;
	}
}

class Topology
{
	constructor(learningRate, size) {
		this.learningRate = learningRate;
		this.layerSize = size;
		this.layerCount = size.length;
	}
}

class Layer
{
	constructor(size, nextLayerSize) {
		this.size = size;
		this.nextLayerSize = nextLayerSize;

		let methodValue = function() {
			return Math.random() * 2.0 - 1.0;
		}

		this.neurons = Utils.createArray(size, methodValue);
		this.biases = Utils.createArray(size, methodValue);

		this.weights = Utils.createTwoWorldArray(size, nextLayerSize, methodValue);
	}
}

class Network
{
	constructor(topology) {
		this.topology = topology;
		this.layers = new Array(topology.layerCount);

		this.iterationCount = 0;

		let layerSize = topology.layerSize;

		for (let i = 0; i < layerSize.length; i++) {
			let nextLayerSize = 0;

			if (i < layerSize.length - 1)
			{
				nextLayerSize = layerSize[i + 1];
			}

			this.layers[i] = new Layer(layerSize[i], nextLayerSize);
		}
	}

	feedForword(inputs) {
		this.layers[0].neurons = inputs;

		for (let i = 1; i < this.layers.length; i++) {
			let currentLayer = this.layers[i - 1];
			let nextLayer = this.layers[i];

			for (let j = 0; j < nextLayer.size; j++) {
				nextLayer.neurons[j] = 0.0;

				for (let k = 0; k < currentLayer.size; k++) {
					nextLayer.neurons[j] += currentLayer.neurons[k] * currentLayer.weights[k][j];
				}

				nextLayer.neurons[j] += nextLayer.biases[j];
				nextLayer.neurons[j] = Utils.sigmoid(nextLayer.neurons[j]);
			}
		}

		return this.layers[this.layers.length - 1].neurons;
	}

	backpropagation(targets, inputs) {
		let networkOutputs = this.feedForword(inputs);
		this.iterationCount += 1;

		let lastLayer = this.layers[this.layers.length - 1];
		let errors = new Array(lastLayer.size);

		for (let i = 0; i < lastLayer.size; i++) {
			errors[i] = targets[i] - lastLayer.neurons[i];
		}

		for (let k = this.layers.length - 2; k >= 0; k--) {
			let currentLayer = this.layers[k];
			let nextLayer = this.layers[k + 1];

			let gradients = new Array(nextLayer.size);
			for (let i = 0; i < gradients.length; i++) {
				gradients[i] = errors[i] * Utils.dsigmoid(nextLayer.neurons[i]) * this.topology.learningRate;
			}

			let deltas = new Array(nextLayer.size);
			for (let i = 0; i < nextLayer.size; i++) {
				deltas[i] = new Array(currentLayer.size);

				for (let j = 0; j < currentLayer.size; j++) {
					deltas[i][j] = gradients[i] * currentLayer.neurons[j];
				}
			}

			let newErrors = new Array(currentLayer.size);
			for (let i = 0; i < currentLayer.size; i++) {
				newErrors[i] = 0.0;

				for (let j = 0; j < nextLayer.size; j++) {
					newErrors[i] += currentLayer.weights[i][j] * errors[j];
				}
			}

			errors = Object.freeze(newErrors);

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

module.exports.Network = Network;
module.exports.Topology = Topology;
module.exports.Utils = Utils;
