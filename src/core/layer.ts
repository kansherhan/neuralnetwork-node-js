class Layer
{
    readonly size: number;
    readonly nextLayerSize: number;

    neurons: Array<number>;
    biases: Array<number>;

    weights: Array<Array<number>>;

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

    static getValueDefault(): number {
        return Math.random() * 2.0 - 1.0;
    }
}

export { Layer };