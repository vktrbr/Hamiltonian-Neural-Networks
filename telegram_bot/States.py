from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup


class StateMain(StatesGroup):
    type_pendulum_sp = State()
    type_pendulum_dp = State()
    q_values_sp = State()
    q_values_dp = State()
    p_values_sp = State()
    p_values_dp = State()
