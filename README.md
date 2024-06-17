# Analysis-of-Tonality
В данном репозитории представлен код к ВКР на тему "Анализ тональности постов в Telegram-каналах на рынке акций с помощью методов машинного обучения"
Веса моделей можно скачать по ссылкам:
  - [Здесь](https://drive.google.com/file/d/19eNlUBAQVhIDHZaFyR9EQrzMuKkqAfw4/view?usp=drive_link) для NER,
  - [Здесь](https://drive.google.com/file/d/1-ABsX5-SoiPar8FcJQ8TEIjlTt7xdin2/view?usp=drive_link) для TSA.

Для корректной работы файлы с весами моделей нужно создать папку models и поместиь туда веса.

Для запуска inference необходимо установить poetry:

``` cmd
  pip install poetry
```

установить окружение:

``` cmd
  poetry install
```

запустить приложение:

``` cmd
  streamlit run app.py
```
