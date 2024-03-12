class Logger:
    logs = []
    log_file_path = 'app.log'

    @staticmethod
    def add_log(level, message):
        formatted_message = f'{level} - {message}'
        print(formatted_message)  # Для вывода в консоль (опционально)
        Logger.logs.append(formatted_message)  # Добавление сообщения в список логов

    @staticmethod
    def info(message):
        Logger.add_log('INFO', message)

    @staticmethod
    def warning(message):
        Logger.add_log('WARNING', message)

    @staticmethod
    def error(message):
        Logger.add_log('ERROR', message)

    @staticmethod
    def exception(message):
        Logger.add_log('EXCEPTION', message)

    @staticmethod
    def clear():
        Logger.logs = []  # Очистка списка логов

    @staticmethod
    def save(file_path=None):
        if file_path is None:
            file_path = Logger.log_file_path

        with open(file_path, 'w') as file:
            file.write('\n'.join(Logger.logs))
