#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <string>
#include <limits>
#include <algorithm>

using namespace std;

struct Edge
{
    string to;
    double weight;
};

unordered_map<string, vector<Edge>> graph;

string edgeKey(string a, string b)
{
    if (a > b)
        swap(a, b);
    return a + "|" + b;
}

void loadGraph(const string &filename)
{
    ifstream file(filename);

    if (!file)
    {
        cerr << "Could not open graph_data.txt" << endl;
        return;
    }

    string from, to;
    double weight;

    while (file >> from >> to >> weight)
    {
        graph[from].push_back({to, weight});
        graph[to].push_back({from, weight});
    }

    file.close();
}

unordered_set<string> loadBlockedEdges(const string &filename)
{
    unordered_set<string> blocked;

    ifstream file(filename);

    if (!file)
    {
        return blocked;
    }

    string a, b;

    while (file >> a >> b)
    {
        blocked.insert(edgeKey(a, b));
    }

    file.close();

    return blocked;
}

void dijkstra(
    const string &start,
    const string &target,
    const unordered_set<string> &blocked)
{
    unordered_map<string, double> distance;
    unordered_map<string, string> previous;

    for (auto &node : graph)
    {
        distance[node.first] =
            numeric_limits<double>::infinity();
    }

    distance[start] = 0;

    priority_queue<
        pair<double, string>,
        vector<pair<double, string>>,
        greater<pair<double, string>>>
        pq;

    pq.push({0, start});

    while (!pq.empty())
    {

        double currentDistance = pq.top().first;
        string current = pq.top().second;

        pq.pop();

        if (currentDistance != distance[current])
        {
            continue;
        }

        if (current == target)
        {
            break;
        }

        for (const Edge &edge : graph[current])
        {

            if (blocked.count(edgeKey(current, edge.to)))
            {
                continue;
            }

            double newDistance =
                currentDistance + edge.weight;

            if (newDistance < distance[edge.to])
            {

                distance[edge.to] = newDistance;
                previous[edge.to] = current;

                pq.push({newDistance,
                         edge.to});
            }
        }
    }

    if (distance[target] ==
        numeric_limits<double>::infinity())
    {

        cout << "NO_ROUTE" << endl;
        return;
    }

    vector<string> path;
    string current = target;

    while (current != start)
    {

        path.push_back(current);
        current = previous[current];
    }

    path.push_back(start);

    reverse(path.begin(), path.end());

    cout << "PATH";

    for (const string &node : path)
    {
        cout << " " << node;
    }

    cout << endl;

    cout << "DISTANCE "
         << distance[target]
         << endl;
}

int main(int argc, char *argv[])
{
    // if (argc < 3)
    // {
    //     cerr << "Usage: ./traffic START DESTINATION" << endl;
    //     return 1;
    // }

    // string start = argv[1];
    // string target = argv[2];

    loadGraph("graph_data.txt");

    // if (graph.find(start) == graph.end())
    // {
    //     cerr << "Invalid start node." << endl;
    //     return 1;
    // }

    // if (graph.find(target) == graph.end())
    // {
    //     cerr << "Invalid destination node." << endl;
    //     return 1;
    // }

    unordered_set<string> blocked =
        loadBlockedEdges("blocked_edges.txt");

    ifstream file("cars.txt");

    if (!file)
    {
        cerr << "Could not open cars.txt" << endl;
        return 1;
    }

    int car;
    string start, target;

    while (file >> car >> start >> target)
    {
        cout << "CAR" << car << endl;
        dijkstra(start, target, blocked);
    }

    file.close();
    // dijkstra(start, target, blocked);

    return 0;
}
