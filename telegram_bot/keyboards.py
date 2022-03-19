from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

btnHello = KeyboardButton('Привет ')
greet_kb = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True).add(btnHello)

# ------ Клавиатура вида маятника ---------
btnSP = KeyboardButton('Single pendulum', )
btnDP = KeyboardButton('Double pendulum')

greetKeyBoardMain = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True).add(btnSP, btnDP)
# ---------------------------------------
