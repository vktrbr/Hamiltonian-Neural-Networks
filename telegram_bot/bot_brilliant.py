import logging

from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Text
from aiogram.utils import executor

import keyboards as kb
from States import StateMain
# --- import from other project files
from config import TOKEN
from create_video import give_video_sp, give_video_dp

logging.basicConfig(level=logging.INFO)
bot = Bot(token=TOKEN)

storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)


@dp.message_handler(commands='start')
async def cmd_start(message: types.Message):
    """
    Conversation's entry point
    """
    await message.answer('Hello! Welcome to Hamiltonian neural network bot!\n'
                         'Use the keyboard to work with the bot. If you need help, click on /help',
                         reply_markup=kb.greetKeyBoardMain)


@dp.message_handler(state='*', commands='cancel')
@dp.message_handler(Text(equals='cancel', ignore_case=True), state='*')
async def cancel_handler(message: types.Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state is None:
        return

    logging.info('Cancelling state %r', current_state)

    await state.finish()
    await message.reply('Main menu', reply_markup=kb.greetKeyBoardMain)


@dp.message_handler(Text(equals='single pendulum', ignore_case=True), state='*')
async def start_settings(message: types.Message, state: FSMContext):
    await StateMain.type_pendulum_sp.set()
    async with state.proxy() as data:
        data['type'] = 'sp'
    await message.answer('Okay. Enter q value (q in [-2, 2]) \n'
                         'For example: \n1.2 ')


@dp.message_handler(Text(equals='double pendulum', ignore_case=True), state='*')
async def start_settings(message: types.Message, state: FSMContext):
    await StateMain.type_pendulum_dp.set()
    async with state.proxy() as data:
        data['type'] = 'dp'
    await message.answer('Okay. Enter comma-separated q values (q in [-1.5, 1.5]) \n'
                         'For example: \n1.2, -0.5')


async def check_invalid_number(message: types.Message):
    try:
        l = message.text.split(',')
        if len(l) != 2:
            return True
        elif len(l) == 2:
            float(l[0]), float(l[1])
        float(l[0])
        return False
    except ValueError:
        return True


async def change_number_to_valid(number: float, valid: float, name: str):
    if number > valid:
        return [False, f'We made {name}={valid}, because your {name}>{valid}', valid]

    elif number < -valid:
        return [False, f'We made {name}={-valid}, because your {name}<{-valid}', -valid]

    return [True, '', number]


async def check_invalid_number_sp(message: types.Message):
    try:
        float(message.text)
        return False
    except Exception as e:
        return True


@dp.message_handler(check_invalid_number_sp, state=StateMain.type_pendulum_sp)
async def process_q(message: types.Message):
    return await message.reply('Error. Please enter the correct value')


@dp.message_handler(check_invalid_number, state=StateMain.type_pendulum_dp)
async def process_q(message: types.Message):
    return await message.reply('Error. Please enter the correct value')


@dp.message_handler(state=StateMain.type_pendulum_sp)
async def process_q(message: types.Message, state: FSMContext):
    q = message.text

    async with state.proxy() as data:
        flag, msg, q = await change_number_to_valid(float(q), 2, 'q')
        data['q'] = q
        if not flag:
            await message.reply(msg)

    await StateMain.q_values_sp.set()
    await message.reply('Next. Enter p value (p in [-5, 5])')


@dp.message_handler(state=StateMain.type_pendulum_dp)
async def process_q(message: types.Message, state: FSMContext):
    async with state.proxy() as data:

        q1, q2 = message.text.split(',')
        flag, msg, q1 = await change_number_to_valid(float(q1), 1.5, 'q1')
        data['q1'] = q1
        if not flag:
            await message.reply(msg)
        flag, msg, q2 = await change_number_to_valid(float(q2), 1.5, 'q2')
        data['q2'] = q2
        if not flag:
            await message.reply(msg)

    await StateMain.q_values_dp.set()
    await message.reply('Next. Enter comma-separated p values (p in [-5, 5])')


@dp.message_handler(check_invalid_number_sp, state=StateMain.q_values_sp)
async def process_q(message: types.Message):
    return await message.reply('Error. Please enter the correct value')


@dp.message_handler(check_invalid_number, state=StateMain.q_values_dp)
async def process_q(message: types.Message):
    return await message.reply('Error. Please enter the correct value')


@dp.message_handler(state=StateMain.q_values_sp)
async def process_q(message: types.Message, state: FSMContext):
    p = message.text

    async with state.proxy() as data:

        if data['type'] == 'sp':
            flag, msg, p = await change_number_to_valid(float(p), 5, 'p')
            data['p'] = p
            if not flag:
                await message.reply(msg)

            await message.answer("Great! There's this video of a single pendulum with hnn predict.\n"
                                 "Wait for 2-5 minutes, please.", reply_markup=kb.greetKeyBoardMain)
            path = 'videos/single_pendulum_' + '_' + str(data['q']) + '_' + str(data['p'])
            path = path.replace('-', 't')
            path = path.replace('.', 'd')

            try:
                with open(path + '.mp4', 'rb') as f:
                    await bot.send_animation(message.chat.id, f)
            except Exception as e:
                path, time_calc, time_vid = give_video_sp(data['q'], data['p'], path)
                with open(path + '.mp4', 'rb') as f:
                    await bot.send_animation(message.chat.id, f)
                    await message.answer(f'The trajectory was calculated in {time_calc :0.4f} seconds \n'
                                         f'The animation was created in {time_vid :0.3f} seconds ',
                                         reply_markup=kb.greetKeyBoardMain)

    await state.finish()


async def check_file(path):
    try:
        with open(path, 'rb') as f:
            return True
    except Exception as e:
        return False


@dp.message_handler(state=StateMain.q_values_dp)
async def process_q(message: types.Message, state: FSMContext):
    p = message.text

    async with state.proxy() as data:

        if data['type'] == 'dp':

            p1, p2 = message.text.split(',')
            flag, msg, p1 = await change_number_to_valid(float(p1), 5, 'p1')
            data['p1'] = p1
            if not flag:
                await message.reply(msg)
            flag, msg, p2 = await change_number_to_valid(float(p2), 5, 'p2')
            data['p2'] = p2
            if not flag:
                await message.reply(msg)

            await message.answer("Great! There's this video of a double pendulum with hnn predict.\n"
                                 "Wait for 2-5 minutes, please.", reply_markup=kb.greetKeyBoardMain)

            path = 'videos/double_pendulum_' + '_' + str(data['q1']) + '_' + str(data['q2']) + '_' + str(
                data['p1']) + '_' + str(data['p2'])
            path = path.replace('-', 't')
            path = path.replace('.', 'd')

            try:
                with open(path + '.mp4', 'rb') as f:
                    await bot.send_animation(message.chat.id, f)
            except Exception as e:
                path, time_calc, time_vid = give_video_dp([data['q1'], data['q2'], data['p1'], data['p2']], path)

                with open(path + '.mp4', 'rb') as f:
                    await bot.send_animation(message.chat.id, f)
                    await message.answer(f'The trajectory was calculated in {time_calc :0.4f} seconds \n'
                                         f'The animation was created in {time_vid :0.3f} seconds ',
                                         reply_markup=kb.greetKeyBoardMain)

    await state.finish()


if __name__ == '__main__':
    executor.start_polling(dp)
