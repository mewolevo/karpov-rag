from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import cohere
import os

# Подключение к API Cohere (или Hugging Face)
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
model = SentenceTransformer('all-MiniLM-L6-v2')


def search_courses(query_embedding, index, courses_data, top_k=5):
    """
    Поиск релевантных курсов по эмбеддингу запроса в FAISS индексе.
    """
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if idx != -1 and idx < len(courses_data):
            results.append(courses_data[idx])
    return results


def load_courses_data(filename="../data/courses_data.json"):
    """
    Загрузка данных о курсах.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        courses_data = json.load(f)
    return courses_data


def load_faiss_index(filename="../data/course_embeddings.index"):
    """
    Загрузка FAISS индекса.
    """
    return faiss.read_index(filename)


def get_query_embedding(query, model_type='cohere'):
    """
    Генерация эмбеддинга для пользовательского запроса.
    """
    if model_type == 'cohere':
        response = cohere_client.embed(model="embed-multilingual-light-v3.0", input_type="search_query", texts=[query])
        return response.embeddings[0]
    elif model_type == 'huggingface':
        return model.encode([query])[0]


def chunk_text(text, chunk_size=200):
    """
    Разбиение текста на чанки заданного размера.
    """
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def get_embeddings(texts):
    """
    Получение эмбеддингов для списка текстов.
    """
    return model.encode(texts)


def save_faiss_index(embeddings, filename="../data/course_embeddings.index"):
    """
    Сохранение эмбеддингов в FAISS индекс.
    """
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)

    embeddings_np = np.array(embeddings).astype('float32')
    index.add(embeddings_np)
    faiss.write_index(index, filename)


def process_courses(courses_data):
    """
    Обработка курсов: чанкование текста, создание эмбеддингов и сохранение индекса.
    """
    all_embeddings = []
    for course in courses_data:
        description = course['description']
        chunks = chunk_text(description)
        embeddings = get_embeddings(chunks)
        all_embeddings.extend(embeddings)

    save_faiss_index(all_embeddings)
    return all_embeddings


def process_user_query(user_query, index, courses_data, llm_available=True):
    """
    Обработка запроса пользователя с учетом данных индекса и LLM.
    """
    try:
        # Генерация эмбеддинга для запроса
        query_embedding = get_query_embedding(user_query)

        # Поиск релевантных курсов в индексе
        relevant_courses = search_courses(query_embedding, index, courses_data)

        # Если LLM доступна, используем ее
        if llm_available:
            try:
                # Генерация ответа с учетом данных курсов
                context = "\n".join([f"Курс: {course['title']}." for course in relevant_courses])
                prompt = f"Вопрос: {user_query}\nКонтекст:\n{context}\nОтвет:"
                response = cohere_client.generate(prompt=prompt, model="command-xlarge-nightly")
                print(response)
                return response.generations[0].text.strip()
            except Exception as e:
                print(f"Ошибка при работе с LLM: {e}")

        # Резервный ответ на основе данных из индекса
        fallback_response = "К сожалению, я не могу обратиться к языковой модели. Вот релевантные курсы:\n"
        for course in relevant_courses:
            fallback_response += f"- {course['title']}: {course['description'][:200]}...\n"
        return fallback_response

    except Exception as e:
        return f"Ошибка обработки запроса: {e}"
