import subprocess
from matplotlib import pyplot as plt

def mean(x):
    return sum(x) / len(x)

program_names = ["kmeans_cuda", "kmeans_mp", "kmeans_seq"]
max_iterations = 100
num_runs = 10
labels = ["CUDA", "OpenMP", "A. sekwencyjny"]

outputs = {
    "kmeans_cuda": [],
    "kmeans_mp": [],
    "kmeans_seq": [],
}

arguments = range(10, max_iterations + 10, 10)
for program_name in program_names:
    print(program_name)
    for arg in arguments:
        print(f'\t./{program_name} {arg}')
        arg_output = []
        for i in range(num_runs):
            print(f'\t\tIteration {i+1}')
            process = subprocess.Popen(["./" + program_name, str(arg)], stdout=subprocess.PIPE)
            output, _ = process.communicate()
            output = float(output.decode().strip())
            arg_output.append(output)
            print(f'\t\t{output}')
        outputs[program_name].append(mean(arg_output))
        print(f'\tMean = {outputs[program_name][-1]}')

for program_name, label in zip(program_names, labels):
    plt.plot(arguments, outputs[program_name], label = label)

plt.xlabel('Ilość iteracji')
plt.ylabel('Czas wykonywania algorytmu [s]')
plt.title('Algorytm k-średnich')
plt.legend()
plt.savefig('test.png')
