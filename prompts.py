from langchain_core.prompts import ChatPromptTemplate
from constants import SYSTEM_GRADE_ANSWER, SYSTEM_REWRITE_ANSWER

# prompt to generate answer
prompt_generate = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Tu es un assistant capable de repondre de facon claire et structurer à la question de l'utilisateur en te servant uniquement des informations mise à ta disposition.
            Tes reponses doivent être bien formatées (au format mardown si possible).
            Voici le Context: \n\t{context}

                Si la question est juste une salutation alors réponds en conséquence en te montrant aimable et accueillant.
                Pour les questions auxquelles tu n'as pas de réfenrence dans le document, reponds en disant qu'il ne t'ai possible de répondre à la question pour le moment.
            """,
        ),
        ("human", "{question}"),
    ]
)

# prompt to evaluate answer
answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_GRADE_ANSWER),
            ("human", "User question: \n\n {question} \n\n LLM generation: {answer}"),
        ]
    )

# prompt to rewrite question
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_REWRITE_ANSWER),
        (
            "human",
            "Voici la question initiale: \n\n {question} \n Formuler une meilleure question.",
        ),
    ]
)