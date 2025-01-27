import requests
from bs4 import BeautifulSoup
import json

# URL страницы с курсами
BASE_URLS = \
    ["https://karpov.courses/analytics",
"https://karpov.courses/analytics-hard",
"https://karpov.courses/ml-hard",
"https://karpov.courses/systemdesign",
"https://karpov.courses/data-driven",
"https://karpov.courses/dataengineer-start",
"https://karpov.courses/deep-learning",
"https://karpov.courses/dataengineer",
"https://karpov.courses/ml-start",
]


def fetch_course_data(url):
    """
    Функция для извлечения данных о курсе с конкретной страницы.
    """
    # Отправляем запрос на страницу
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Ошибка при запросе: {response.status_code} для URL: {url}")
        return None

    # Парсим HTML-страницу
    soup = BeautifulSoup(response.text, 'html.parser')

    # Извлекаем title (название курса)
    title = soup.find('title').get_text(strip=True)
    print(f"Title: {title} для URL: {url}")  # Выводим title для дебага

    # Извлекаем текст из всех элементов с классом tn-atom
    tn_atoms = soup.find_all('div', class_='tn-atom')
    description = " ".join([tn_atom.get_text(strip=True) for tn_atom in tn_atoms])

    print(f"Description: {description[:100]}...")  # Выводим первые 100 символов описания для дебага

    # Формируем данные для курса
    course_data = {
        'title': title,
        'description': description,
        'url': url
    }

    return course_data


def fetch_all_courses():
    """
    Функция для извлечения данных о курсах с нескольких страниц.
    """
    courses = []

    # Проходим по всем URL и собираем данные
    for url in BASE_URLS:
        course_data = fetch_course_data(url)
        if course_data:
            courses.append(course_data)

    return courses


def save_courses(courses, filename="courses_data.json"):
    """
    Сохраняет данные о курсах в файл JSON.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(courses, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # Собираем данные с нескольких сайтов
    courses = fetch_all_courses()
    if courses:
        save_courses(courses)
        print(f"Найдено {len(courses)} курсов. Данные сохранены.")
    else:
        print("Курсы не были найдены.")