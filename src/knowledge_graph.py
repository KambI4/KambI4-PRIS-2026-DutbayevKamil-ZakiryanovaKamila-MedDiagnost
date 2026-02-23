from models import Disease


def create_graph(diseases=None):
    graph = nx.Graph()

    if diseases is None:
        diseases = [
            Disease("Грипп", ["Температура", "Кашель", "Слабость"], ["Парацетамол"]),
            Disease("COVID-19", ["Температура", "Головная боль", "Слабость"], ["Противовирусные"]),
            Disease("Простуда", ["Кашель", "Слабость"], ["Ибупрофен"]),
        ]

    symptom_nodes = sorted({symptom for d in diseases for symptom in d.symptoms})
    medicine_nodes = sorted({medicine for d in diseases for medicine in d.medicines})

    graph.add_nodes_from((d.name for d in diseases), type="disease")
    graph.add_nodes_from(symptom_nodes, type="symptom")
    graph.add_nodes_from(medicine_nodes, type="medicine")

    relationships = []
    for disease in diseases:
        relationships.extend((disease.name, symptom) for symptom in disease.symptoms)
        relationships.extend((disease.name, medicine) for medicine in disease.medicines)

    graph.add_edges_from(relationships)
    return graph


def load_graph():
    return create_graph()


def find_related_entities(graph, node):
    if node not in graph:
        return []
    return list(graph.neighbors(node))