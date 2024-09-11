Um das **AI Scientist**-Projekt zu starten с использованием всех игнорируемых промптов из `README.md`, вы можете использовать следующую команду в терминале. Предполагается, что вы находитесь в директории `/Domain` и что у вас установлен Docker.

### Команда для запуска AI Scientist

```bash
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -v /Domain/templates:/app/AI-Scientist/templates <AI_SCIENTIST_IMAGE> \
--model gpt-4o-2024-05-13 \
--experiment 2d_diffusion \
--num-ideas 2 \
--ignore-prompts
```

### Объяснение команды:

- **`docker run`**: Команда для запуска контейнера Docker.
- **`-e OPENAI_API_KEY=$OPENAI_API_KEY`**: Устанавливает переменную окружения для API-ключа OpenAI.
- **`-v /Domain/templates:/app/AI-Scientist/templates`**: Монтирует локальную директорию с шаблонами в контейнер.
- **`<AI_SCIENTIST_IMAGE>`**: Замените это на имя вашего Docker-образа AI Scientist.
- **`--model gpt-4o-2024-05-13`**: Указывает модель, которую вы хотите использовать.
- **`--experiment 2d_diffusion`**: Указывает эксперимент, который вы хотите запустить.
- **`--num-ideas 2`**: Указывает количество идей, которые вы хотите сгенерировать.
- **`--ignore-prompts`**: Указывает, что вы хотите игнорировать определенные промпты, как указано в `README.md`.

### Примечание

Убедитесь, что вы заменили `<AI_SCIENTIST_IMAGE>` на фактическое имя образа Docker, который вы используете для AI Scientist. Если у вас есть дополнительные параметры или настройки, которые нужно учесть, вы можете добавить их в команду.

docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -v /Domain/templates:/app/AI-Scientist/templates <AI_SCIENTIST_IMAGE> --model gpt-4o --experiment 2d_diffusion --num-ideas 2 --ignore
