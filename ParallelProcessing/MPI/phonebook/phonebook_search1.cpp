// //mpic++ phonebook_search1.cpp -o p
// //mpirun -np 4 ./p phonebook1.txt AISHWARYA


#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

/* ================= MPI STRING SEND / RECEIVE ================= */

void send_string(const string &text, int dest) {
    int len = text.size() + 1;
    MPI_Send(&len, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
    MPI_Send(text.c_str(), len, MPI_CHAR, dest, 0, MPI_COMM_WORLD);
}

string receive_string(int src) {
    MPI_Status status;
    int len;
    MPI_Recv(&len, 1, MPI_INT, src, 0, MPI_COMM_WORLD, &status);
    vector<char> buf(len);
    MPI_Recv(buf.data(), len, MPI_CHAR, src, 0, MPI_COMM_WORLD, &status);
    return string(buf.data());
}

/* ================= UTILITY FUNCTIONS ================= */

string to_lower(const string &s) {
    string t = s;
    transform(t.begin(), t.end(), t.begin(), ::tolower);
    return t;
}

string vector_to_string(const vector<string> &v, int start, int end) {
    string result;
    for (int i = start; i < min((int)v.size(), end); i++) {
        result += v[i] + "\n";
    }
    return result;
}

vector<string> string_to_vector(const string &s) {
    vector<string> lines;
    string line;
    istringstream iss(s);
    while (getline(iss, line)) {
        if (!line.empty())
            lines.push_back(line);
    }
    return lines;
}

bool match(const string &line, const string &key) {
    return to_lower(line).find(to_lower(key)) != string::npos;
}

/* ================= FILE READ ================= */

void read_files(const vector<string> &files, vector<string> &lines) {
    for (auto &file : files) {
        ifstream fin(file);
        if (!fin.is_open()) {
            cerr << "❌ Cannot open file: " << file << endl;
            continue;
        }
        string line;
        while (getline(fin, line)) {
            if (!line.empty())
                lines.push_back(line);
        }
        fin.close();
    }
}

/* ================= MAIN ================= */

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) {
            cerr << "Usage:\n";
            cerr << "mpirun -np <procs> ./p <file1> <file2> ... <search_term>\n";
        }
        MPI_Finalize();
        return 1;
    }

    string search_key = argv[argc - 1];
    double t1, t2;

    /* ================= MASTER ================= */
    if (rank == 0) {
        vector<string> files;
        for (int i = 1; i < argc - 1; i++)
            files.push_back(argv[i]);

        vector<string> all_lines;
        read_files(files, all_lines);

        cout << "Total lines read: " << all_lines.size() << endl;

        int total = all_lines.size();
        int chunk = (total + size - 1) / size;

        /* Send chunks */
        for (int i = 1; i < size; i++) {
            string part = vector_to_string(all_lines, i * chunk, (i + 1) * chunk);
            send_string(part, i);
        }

        /* Master search */
        t1 = MPI_Wtime();
        string final_result;

        for (int i = 0; i < min(chunk, total); i++) {
            if (match(all_lines[i], search_key))
                final_result += all_lines[i] + "\n";
        }

        /* Collect results */
        for (int i = 1; i < size; i++) {
            final_result += receive_string(i);
        }
        t2 = MPI_Wtime();

        ofstream fout("output.txt", ios::out | ios::trunc);
        fout << final_result;
        fout.close();

        cout << "✅ Search completed\n";
        cout << "⏱ Time: " << t2 - t1 << " seconds\n";
        cout << "📁 Output written to output.txt\n";
    }

    /* ================= WORKER ================= */
    else {
        string recv_data = receive_string(0);
        vector<string> lines = string_to_vector(recv_data);

        t1 = MPI_Wtime();
        string local_result;

        for (auto &line : lines) {
            if (match(line, search_key))
                local_result += line + "\n";
        }
        t2 = MPI_Wtime();

        send_string(local_result, 0);
        cout << "Process " << rank
             << " done in " << t2 - t1 << " sec\n";
    }

    MPI_Finalize();
    return 0;
}
