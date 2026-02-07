%%writefile search_phonebook.cu
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cuda_runtime.h>

using namespace std;

// ================================
// Contact struct
// ================================
struct Contact {
    char name[50];
    char number[50];
};

// ================================
// Device function: substring check
// ================================
__device__ bool isSubstring(const char* text, const char* pattern) {
    for (int i = 0; text[i] != '\0'; i++) {
        int j = 0;
        while (text[i+j] != '\0' && pattern[j] != '\0' && text[i+j] == pattern[j]) {
            j++;
        }
        if (pattern[j] == '\0') return true;
    }
    return false;
}

// ================================
// Kernel: search phonebook (name OR number)
// ================================
__global__ void searchPhonebook(Contact* d_contacts, int num_contacts,
                                char* d_pattern, int* d_results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_contacts) return;

    if (isSubstring(d_contacts[idx].name, d_pattern) ||
        isSubstring(d_contacts[idx].number, d_pattern)) {
        d_results[idx] = 1; // match
    } else {
        d_results[idx] = 0;
    }
}

// ================================
// Main
// ================================
int main(int argc, char* argv[]) {

    if (argc != 3) {
        cout << "Usage: ./search_phonebook <search_pattern> <threads>\n";
        return 0;
    }

    string search_pattern = argv[1];
    int threads = atoi(argv[2]);

    // ================================
    // Read phonebook file
    // ================================
    string file_name = "phonebook1.txt"; // change path if needed
    ifstream file(file_name);
    if (!file) {
        cerr << "Error opening file: " << file_name << endl;
        return 1;
    }

    vector<Contact> contacts;
    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;

        Contact c;
        int pos = line.find(",");
        if (pos != string::npos) {
            // Remove quotes
            string name = line.substr(1, pos - 2);
            string number = line.substr(pos + 2, line.size() - pos - 3);

            strncpy(c.name, name.c_str(), sizeof(c.name)-1);
            c.name[sizeof(c.name)-1] = '\0';
            strncpy(c.number, number.c_str(), sizeof(c.number)-1);
            c.number[sizeof(c.number)-1] = '\0';

            contacts.push_back(c);
        }
    }
    file.close();

    int n = contacts.size();
    cout << "Total contacts: " << n << endl;

    // ================================
    // Allocate device memory
    // ================================
    Contact* d_contacts;
    char* d_pattern;
    int* d_results;

    cudaMalloc(&d_contacts, n * sizeof(Contact));
    cudaMalloc(&d_pattern, search_pattern.size() + 1);
    cudaMalloc(&d_results, n * sizeof(int));

    cudaMemcpy(d_contacts, contacts.data(), n * sizeof(Contact), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pattern, search_pattern.c_str(), search_pattern.size() + 1, cudaMemcpyHostToDevice);

    // ================================
    // Launch kernel
    // ================================
    int blocks = (n + threads - 1) / threads;
    searchPhonebook<<<blocks, threads>>>(d_contacts, n, d_pattern, d_results);

    // Error check
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA Error: " << cudaGetErrorString(err) << endl;
        return 1;
    }

    cudaDeviceSynchronize();

    // ================================
    // Copy results back
    // ================================
    int* h_results = (int*)malloc(n * sizeof(int));
    cudaMemcpy(h_results, d_results, n * sizeof(int), cudaMemcpyDeviceToHost);

    // ================================
    // Print matches
    // ================================
    cout << "Matched contacts:\n";
    for (int i = 0; i < n; i++) {
        if (h_results[i]) {
            cout << contacts[i].name << " , " << contacts[i].number << endl;
        }
    }

    // ================================
    // Cleanup
    // ================================
    cudaFree(d_contacts);
    cudaFree(d_pattern);
    cudaFree(d_results);
    free(h_results);

    return 0;
}


//!nvcc -arch=sm_75 search_phonebook.cu -o search_phonebook
//!time ./search_phonebook AKTER 100 > output1.txt