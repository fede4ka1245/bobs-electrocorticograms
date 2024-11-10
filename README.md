# Документация

## 🚀 Запуск решения

1) Нужно скопировать содержимое файла .env.example в файл .env
2) Убедитесь что у вас установлен docker. Запустите в корне проекта следующую команду: 
```
docker compose up -d
```
У вас поднимутся два контейнера:
- Streamlit. Сервис для генерации разметки (http://localhost:8501)
- Label Studio. Сервис для разметки данных (http://localhost:8080)

*При первом запуске может быть долгая загрузка, так как нужно скачать все зависимости. Может потребоваться перезапуск, если упадет timeout*

## 🤖 ML-solution
Инструкция по работе с ML-решением: [/ml-solution/README.md](https://github.com/fede4ka1245/bobs-electrocorticograms/tree/main/ml-solution)

## 🔮 Predictions
Предикты на тестовые данные находятся тут: [/predictions](https://github.com/fede4ka1245/bobs-electrocorticograms/tree/main/predictions)
