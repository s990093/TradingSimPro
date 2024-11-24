class Action:
    def execute(self) -> str:
        """執行動作並返回字符串"""
        raise NotImplementedError

class SkipAction(Action):
    def execute(self) -> str:
        return 'SKIP'

class EnterAction(Action):
    def __init__(self, position_size: float = 1):
        self.position_size = position_size

    def execute(self) -> str:
        return 'ENTER'

class ReduceAction(Action):
    def __init__(self, new_position: int):
        self.new_position = new_position

    def execute(self) -> str:
        return f'REDUCE to {self.new_position}'

class ExitAction(Action):
    def execute(self) -> str:
        return 'EXIT'

class AddPositionAction(Action):
    def __init__(self, new_position: int):
        self.new_position = new_position

    def execute(self) -> str:
        return f'ADD to {self.new_position}'
