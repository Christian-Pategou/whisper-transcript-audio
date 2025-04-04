from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from constants import SYSTEM_GRADE_DOCUMENT, SYSTEM_ANSWER_QUESTION, SYSTEM_GRADE_ANSWER, SYSTEM_REWRITE_ANSWER
from typing import Literal, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_groq.chat_models import ChatGroq
from langchain_core.runnables.base import RunnableSequence
from langchain_core.output_parsers import StrOutputParser

class GradeDocuments(BaseModel):
    """Score binaire pour la vérification de la pertinence des documents récupérés."""

    binary_score: Literal["oui", "non"] = Field(
        description="Si les documents sont pertinents pour la question, 'oui' ou 'non'"
    )

class GradeAnswer(BaseModel):
    """Score binaire pour évaluer la réponse à la question."""

    answer_score: Literal["bon", "mauvais", "humain"] = Field(
        description="La réponse à la question est 'bon', 'mauvais' ou 'humain."
    )


def chain_retrieval_grader_document(model: Union[ChatGoogleGenerativeAI, ChatGroq, BaseChatModel]) -> RunnableSequence:
    """
        Fonction pour renvoyer le chain pour evaluer la perfromence d'un document.
        Agrs:
            model: model de chat(llm)
        Return:
            runnable
    """

    structured_llm_grader = model.with_structured_output(GradeDocuments)

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_GRADE_DOCUMENT),
            ("human", "Documents recupérés: \n\n {document} \n\n Question de l'utilisateur: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader



def chain_asnwer_question(model: Union[ChatGoogleGenerativeAI, ChatGroq, BaseChatModel]) -> RunnableSequence:
    """
        Chain pour repondre à la question en fonction du context fourni.
        Agrs:
            model: model de chat(llm)
        Return:
            runnable
    """
    prompt_generate = ChatPromptTemplate.from_messages(
        [
            (
                "system", SYSTEM_ANSWER_QUESTION,
            ),
            (
                "human", 
                "{question}"
            ),
        ]
    )

    rag_chain = prompt_generate | model | StrOutputParser()
    return rag_chain



def chain_grader_question_answer(model: Union[ChatGoogleGenerativeAI, ChatGroq, BaseChatModel]) -> RunnableSequence:


    structured_llm_grader = model.with_structured_output(GradeAnswer)

    # Prompt
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_GRADE_ANSWER),
            ("human", "User question: \n\n {question} \n\n LLM generation: {answer}"),
        ]
    )
    answer_grader = answer_prompt | structured_llm_grader
    return answer_grader

def chain_rewriter_question(model:Union[ChatGoogleGenerativeAI, ChatGroq, BaseChatModel]) -> RunnableSequence:
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_REWRITE_ANSWER),
            (
                "human",
                "Voici la question initiale: \n\n {question} \n Formuler une meilleure question.",
            ),
        ]
    )
    re_writer_chain = re_write_prompt | model | StrOutputParser()
    return re_writer_chain