from langchain.schema import Document

def get_extra_docs():
    return [
        Document(
            page_content="Dolo 650 is a paracetamol tablet used for fever.",
            metadata={"source": "Doctor"}
        ),
        Document(
            page_content="ORS solution helps prevent dehydration during diarrhea.",
            metadata={"source": "HealthTips"}
        )
    ]
