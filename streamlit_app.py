
import os
import streamlit as st
from llm.embedder import get_query_embedding, load_courses_data, load_faiss_index, search_courses

import cohere
cohere_client = cohere.ClientV2(os.getenv("COHERE_API_KEY"))

def main():
    st.title("Поиск курса на karpov.courses")

    # Описание
    st.write("""
        Здравствуй, путник в мире науки о данных. Мы поможем тебе найти самый подходящий вариант для тебя, оставь свои волнения и введи чего душа желает.
    """)

    st.image("data/images/witch_cat.png", use_container_width=True)

    st.markdown("""
        <style>
            .css-1v3fvcr {
                background-color: #f4e1d2; /* Soft peach background */
            }
            .css-1h6w3bs {
                color: #5c4033; /* Dark brown text */
            }
        </style>
    """, unsafe_allow_html=True)

    # Поле ввода запроса
    query = st.text_input("Что бы вы хотели изучить?", "")

    if query:
        # Загрузка FAISS индекса и данных о курсах
        index = load_faiss_index("llm/course_embeddings.index")
        courses_data = load_courses_data("llm/courses_data.json")

        # Генерация эмбеддинга для запроса
 #       query_embedding = get_query_embedding(query, model_type='cohere')

        # Поиск курсов
        recommendations = process_user_query(query, index, courses_data)

        # Вывод рекомендаций
        if recommendations:
            st.write(recommendations)
        else:
            st.write("Не найдено подходящих курсов.")


def process_user_query(user_query, index, courses_data, llm_available=True, llm_client=cohere_client):
    """
    Обрабатывает пользовательский запрос:
    1. Ищет ближайшие курсы в индексе.
    2. Генерирует ответ с помощью LLM (если доступно).
    3. Если LLM недоступно, возвращает только информацию из индекса.

    Args:
        user_query (str): Запрос пользователя.
        index (faiss.IndexFlatL2): FAISS индекс эмбеддингов курсов.
        courses_data (list): Данные курсов (загруженные из JSON).
        llm_available (bool): Флаг доступности LLM.
        llm_client: Клиент LLM API (например, OpenAI или Cohere).

    Returns:
        str: Ответ для пользователя.
    """
    # Генерация эмбеддинга запроса
    query_embedding = get_query_embedding(user_query)

    # Поиск ближайших курсов в индексе
    top_courses = search_courses(query_embedding, index, courses_data, top_k=3)

    if llm_available and llm_client:
        try:
            # Формируем контекст для LLM
            context = "Вот несколько подходящих курсов:\n" + "\n".join(
                [f"- {course['title']}" for course in top_courses]
            )

            # Запрос к LLM
            llm_prompt = (
                f"Пользователь задал вопрос: '{user_query}'.\n"
                f"{context}\n"
                "Порекомендуй ему курс и кратко объясни, почему он подходит."
            )

            llm_response = llm_client.chat(model="command-r-plus-08-2024",
    messages=[{"role": "user", "content": context}],)

            return llm_response.message.content[0].text
        except Exception as e:
            # Если LLM недоступен, переходим на fallback
            print(f"Ошибка при вызове LLM: {e}")

    # Fallback: только данные из индекса
    fallback_response = "Мы рекомендуем следующие курсы:\n" + "\n".join(
        [f"- {course['title']}" for course in top_courses]
    )
    return fallback_response


if __name__ == "__main__":
    main()
