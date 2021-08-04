import { Network, Topology } from "../neuralnetwork";

const outputs = [ 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1 ];
const inputs = [
    [ 0, 0, 0, 0 ],
    [ 0, 0, 0, 1 ],
    [ 0, 0, 1, 0 ],
    [ 0, 0, 1, 1 ],
    [ 0, 1, 0, 0 ],
    [ 0, 1, 0, 1 ],
    [ 0, 1, 1, 0 ],
    [ 0, 1, 1, 1 ],
    [ 1, 0, 0, 0 ],
    [ 1, 0, 0, 1 ],
    [ 1, 0, 1, 0 ],
    [ 1, 0, 1, 1 ],
    [ 1, 1, 0, 0 ],
    [ 1, 1, 0, 1 ],
    [ 1, 1, 1, 0 ],
    [ 1, 1, 1, 1 ]
];

const topology = new Topology(0.1, [4, 2, 1]);
const network = new Network(topology);

const iterationCount = 5000;
const testCount = 10;

let errorCount = 0;

for (let i = 0; i < iterationCount; i++) {
    const index = Math.abs(Math.round(Math.random() * outputs.length - 1));

    const input = inputs[index];
    const target = outputs[index];

    const networkOutput = network.backpropagation([target], input);

    if (Math.round(networkOutput[0]) != target)
    {
        errorCount += 1;
    }
}

console.log(`Error: ${errorCount}/${iterationCount}`);

console.log("\nTests:");

for (let i = 0; i < testCount; i++) {
    const index = Math.abs(Math.round(Math.random() * outputs.length - 1));

    const input = inputs[index];
    const target = outputs[index];

    const networkOutput = network.feedForword(input);

    const output = networkOutput[0].toString().substr(0, 5);
    console.log(`Target: ${target}, network: ${output}`);
}
