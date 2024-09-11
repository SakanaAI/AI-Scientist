Um die `files-to-prompt`-Funktion des **AI Scientist**-Projekts zu starten с использованием всех игнорируемых промптов из README, вы можете использовать следующую команду в терминале. Предполагается, что вы находитесь в директории `/Users/Lordof44/Downloads/AI-Scientist` и что у вас установлен Docker.

### Команда для запуска

```bash
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -v /Users/Lordof44/Downloads/AI-Scientist/templates:/app/AI-Scientist/templates <AI_SCIENTIST_IMAGE> --model gpt-4o-2024-05-13 --experiment files-to-prompt --ignore-prompts
```

### Объяснение команды:

- **`docker run`**: Команда для запуска контейнера Docker.
- **`-e OPENAI_API_KEY=$OPENAI_API_KEY`**: Устанавливает переменную окружения для API-ключа OpenAI.
- **`-v /Users/Lordof44/Downloads/AI-Scientist/templates:/app/AI-Scientist/templates`**: Монтирует локальную директорию с шаблонами в контейнер.
- **`<AI_SCIENTIST_IMAGE>`**: Замените это на имя вашего Docker-образа AI Scientist.
- **`--model gpt-4o-2024-05-13`**: Указывает модель, которую вы хотите использовать.
- **`--experiment files-to-prompt`**: Указывает эксперимент, который вы хотите запустить.
- **`--ignore-prompts`**: Указывает, что вы хотите игнорировать определенные промпты, как указано в README.

### Примечание

Убедитесь, что вы заменили `<AI_SCIENTIST_IMAGE>` на фактическое имя образа Docker, который вы используете для AI Scientist. Если у вас есть дополнительные параметры или настройки, которые нужно учесть, вы можете добавить их в команду.
