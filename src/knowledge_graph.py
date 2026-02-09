# src/knowledge_graph.py
import networkx as nx

def create_graph():
    G = nx.Graph()

    # --- Узлы ---
    diseases = ["Flu", "Covid", "Cold"]
    symptoms = ["Fever", "Cough", "Headache"]
    medicines = ["Paracetamol", "Ibuprofen"]

    G.add_nodes_from(diseases, type="disease")
    G.add_nodes_from(symptoms, type="symptom")
    G.add_nodes_from(medicines, type="medicine")

    # --- Связи ---
    relationships = [
        ("Flu", "Fever"),
        ("Flu", "Cough"),
        ("Flu", "Paracetamol"),

        ("Covid", "Fever"),
        ("Covid", "Headache"),

        ("Cold", "Cough"),
        ("Cold", "Ibuprofen")
    ]

    G.add_edges_from(relationships)

    return G


def find_related_entities(graph, node):
    if node not in graph:
        return []
    return list(graph.neighbors(node))
