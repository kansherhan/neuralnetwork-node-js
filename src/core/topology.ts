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

export { Topology };