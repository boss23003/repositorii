"""
КОСМИЧЕСКИЙ ЗАЩИТНИК
Стратегия/RPG игра о защите космической станции от вторжения инопланетян
"""

# Основные библиотеки для игрового движка
import pygame
import sys
import random
import time
import math
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union, Callable
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import logging
import json
import os
import pickle
import uuid
import threading
import asyncio
import concurrent.futures

# Библиотеки для графики и анимации
from pygame import gfxdraw
import OpenGL.GL as gl
import OpenGL.GLU as glu
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont

# Библиотеки для обработки звука
import pygame.mixer
from pydub import AudioSegment
import wave
import pyaudio

# Библиотеки для работы с базами данных
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Библиотеки для создания ИИ противников
import tensorflow as tf
import keras
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import gym

# Библиотеки для работы с сетью и многопользовательским режимом
import socket
import requests
import websockets
import aiohttp
import asyncio

# Библиотеки для интерфейса пользователя
import tkinter as tk
from tkinter import ttk
import pyglet
import pygame_gui

# Библиотеки для генерации процедурного контента
import noise
import opensimplex
import voronoi

# Библиотеки для обработки и анализа данных
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Библиотеки для работы с файловой системой и конфигурацией
import configparser
import yaml
import toml
import xml.etree.ElementTree as ET

# Библиотеки для логирования и отладки
import logging.handlers
import traceback
import debugpy

# Дополнительные библиотеки
import arrow  # для работы с датами и временем
import colorama  # для цветного вывода в консоль
import tqdm  # для отображения прогресс-баров
import click  # для создания CLI
import rich  # для улучшенного вывода в терминал
import pyfiglet  # для ASCII-арта

# Инициализация pygame
pygame.init()
pygame.mixer.init()

# Константы
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60
TITLE = "Космический Защитник"

# Цвета
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)

# Настройка базы данных
Base = declarative_base()
engine = create_engine('sqlite:///space_defender.db')
Session = sessionmaker(bind=engine)

# Модель данных для сохранения игры
class PlayerData(Base):
    __tablename__ = 'players'
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    level = Column(Integer)
    experience = Column(Integer)
    credits = Column(Integer)
    station_health = Column(Integer)
    
    upgrades = relationship("UpgradeData", back_populates="player")
    weapons = relationship("WeaponData", back_populates="player")

class UpgradeData(Base):
    __tablename__ = 'upgrades'
    
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey('players.id'))
    name = Column(String)
    level = Column(Integer)
    cost = Column(Integer)
    
    player = relationship("PlayerData", back_populates="upgrades")

class WeaponData(Base):
    __tablename__ = 'weapons'
    
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey('players.id'))
    name = Column(String)
    damage = Column(Integer)
    cooldown = Column(Float)
    
    player = relationship("PlayerData", back_populates="weapons")

# Создание таблиц в базе данных
Base.metadata.create_all(engine)

# Настройка логирования
logger = logging.getLogger('space_defender')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('space_defender.log')
file_handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Классы для игровых объектов
class GameObject(ABC):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.hitbox = pygame.Rect(x, y, width, height)
        self.active = True
        self.id = str(uuid.uuid4())
    
    def update_hitbox(self):
        self.hitbox = pygame.Rect(self.x, self.y, self.width, self.height)
    
    @abstractmethod
    def update(self, dt):
        pass
    
    @abstractmethod
    def draw(self, surface):
        pass
    
    def collides_with(self, other):
        return self.hitbox.colliderect(other.hitbox)

class SpaceStation(GameObject):
    def __init__(self, x, y):
        super().__init__(x, y, 120, 120)
        self.health = 1000
        self.max_health = 1000
        self.shield = 500
        self.max_shield = 500
        self.credits = 1000
        self.weapons = []
        self.crew = []
        self.upgrades = {}
        self.rotation = 0
        self.image = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.image.fill((0, 0, 0, 0))
        pygame.draw.circle(self.image, BLUE, (self.width // 2, self.height // 2), self.width // 2)
        pygame.draw.circle(self.image, CYAN, (self.width // 2, self.height // 2), self.width // 3)
    
    def update(self, dt):
        self.rotation += 0.5 * dt
        self.shield = min(self.shield + 0.1 * dt, self.max_shield)
        self.update_hitbox()
    
    def draw(self, surface):
        rotated_image = pygame.transform.rotate(self.image, self.rotation)
        new_rect = rotated_image.get_rect(center=self.image.get_rect(topleft=(self.x, self.y)).center)
        surface.blit(rotated_image, new_rect.topleft)
        
        # Отображение полосок здоровья и щита
        pygame.draw.rect(surface, RED, (self.x, self.y - 20, self.width, 5))
        health_width = max(0, (self.health / self.max_health) * self.width)
        pygame.draw.rect(surface, GREEN, (self.x, self.y - 20, health_width, 5))
        
        pygame.draw.rect(surface, BLUE, (self.x, self.y - 10, self.width, 5))
        shield_width = max(0, (self.shield / self.max_shield) * self.width)
        pygame.draw.rect(surface, CYAN, (self.x, self.y - 10, shield_width, 5))
    
    def take_damage(self, amount):
        if self.shield > 0:
            shield_damage = min(self.shield, amount)
            self.shield -= shield_damage
            amount -= shield_damage
        
        if amount > 0:
            self.health -= amount
            if self.health <= 0:
                self.health = 0
                return True  # Станция уничтожена
        return False

class Enemy(GameObject):
    def __init__(self, x, y, enemy_type):
        size = random.randint(20, 50)
        super().__init__(x, y, size, size)
        self.enemy_type = enemy_type
        self.health = 50 + enemy_type * 25
        self.max_health = self.health
        self.speed = 1 + enemy_type * 0.5
        self.damage = 10 + enemy_type * 5
        self.attack_cooldown = 0
        self.worth = 10 + enemy_type * 5
        
        # Различный цвет для разных типов врагов
        colors = {
            0: RED,
            1: MAGENTA,
            2: YELLOW,
            3: (255, 128, 0)  # Оранжевый
        }
        self.color = colors.get(enemy_type, RED)
        
        # Создание изображения врага
        self.image = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.image.fill((0, 0, 0, 0))
        
        if enemy_type == 0:  # Scout - треугольник
            points = [
                (self.width // 2, 0),
                (0, self.height),
                (self.width, self.height)
            ]
            pygame.draw.polygon(self.image, self.color, points)
        elif enemy_type == 1:  # Fighter - ромб
            points = [
                (self.width // 2, 0),
                (0, self.height // 2),
                (self.width // 2, self.height),
                (self.width, self.height // 2)
            ]
            pygame.draw.polygon(self.image, self.color, points)
        elif enemy_type == 2:  # Cruiser - шестиугольник
            points = []
            for i in range(6):
                angle = math.pi / 3 * i
                x = self.width // 2 + int(self.width // 2 * math.cos(angle))
                y = self.height // 2 + int(self.height // 2 * math.sin(angle))
                points.append((x, y))
            pygame.draw.polygon(self.image, self.color, points)
        else:  # Dreadnought - восьмиугольник
            points = []
            for i in range(8):
                angle = math.pi / 4 * i
                x = self.width // 2 + int(self.width // 2 * math.cos(angle))
                y = self.height // 2 + int(self.height // 2 * math.sin(angle))
                points.append((x, y))
            pygame.draw.polygon(self.image, self.color, points)
        
        # Добавление свечения с помощью альфа-канала
        glow = pygame.Surface((self.width + 10, self.height + 10), pygame.SRCALPHA)
        for i in range(5):
            alpha = 100 - i * 20
            glow_color = (*self.color, alpha)
            pygame.draw.rect(glow, glow_color, (i, i, self.width + 10 - i*2, self.height + 10 - i*2), 1)
        
        self.image = glow
    
    def update(self, dt, target):
        # Движение к цели
        dx = target.x + target.width // 2 - (self.x + self.width // 2)
        dy = target.y + target.height // 2 - (self.y + self.height // 2)
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance > 0:
            dx /= distance
            dy /= distance
            
            self.x += dx * self.speed * dt
            self.y += dy * self.speed * dt
        
        # Обновление атаки
        if self.attack_cooldown > 0:
            self.attack_cooldown -= dt
        
        self.update_hitbox()
        
        # Проверка столкновения с целью
        return self.collides_with(target)
    
    def draw(self, surface):
        surface.blit(self.image, (self.x, self.y))
        
        # Отображение полоски здоровья
        pygame.draw.rect(surface, RED, (self.x, self.y - 10, self.width, 5))
        health_width = max(0, (self.health / self.max_health) * self.width)
        pygame.draw.rect(surface, GREEN, (self.x, self.y - 10, health_width, 5))
    
    def take_damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            self.health = 0
            return True  # Враг уничтожен
        return False

class Weapon(GameObject):
    def __init__(self, x, y, weapon_type, parent):
        size = 30
        super().__init__(x, y, size, size)
        self.weapon_type = weapon_type
        self.parent = parent
        self.angle = 0
        self.target = None
        self.cooldown = 0
        
        # Параметры оружия в зависимости от типа
        if weapon_type == 0:  # Laser
            self.damage = 10
            self.max_cooldown = 0.5
            self.range = 300
            self.color = RED
        elif weapon_type == 1:  # Plasma
            self.damage = 20
            self.max_cooldown = 1.0
            self.range = 250
            self.color = GREEN
        elif weapon_type == 2:  # Ion
            self.damage = 15
            self.max_cooldown = 0.8
            self.range = 350
            self.color = BLUE
        else:  # Missile
            self.damage = 30
            self.max_cooldown = 2.0
            self.range = 400
            self.color = YELLOW
        
        # Создание изображения оружия
        self.image = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.image.fill((0, 0, 0, 0))
        pygame.draw.circle(self.image, WHITE, (self.width // 2, self.height // 2), self.width // 2)
        pygame.draw.circle(self.image, self.color, (self.width // 2, self.height // 2), self.width // 3)
        
        # Список активных снарядов
        self.projectiles = []
    
    def update(self, dt, enemies):
        # Следование за родительским объектом
        self.x = self.parent.x + self.parent.width // 2 - self.width // 2
        self.y = self.parent.y + self.parent.height // 2 - self.height // 2
        
        # Обновление кулдауна
        if self.cooldown > 0:
            self.cooldown -= dt
        
        # Поиск ближайшего врага в радиусе действия
        nearest_enemy = None
        min_distance = float('inf')
        
        for enemy in enemies:
            dx = enemy.x + enemy.width // 2 - (self.x + self.width // 2)
            dy = enemy.y + enemy.height // 2 - (self.y + self.height // 2)
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance <= self.range and distance < min_distance:
                min_distance = distance
                nearest_enemy = enemy
        
        self.target = nearest_enemy
        
        # Поворот к цели
        if self.target:
            dx = self.target.x + self.target.width // 2 - (self.x + self.width // 2)
            dy = self.target.y + self.target.height // 2 - (self.y + self.height // 2)
            self.angle = math.degrees(math.atan2(-dy, dx))
            
            # Стрельба по цели
            if self.cooldown <= 0:
                self.fire()
                self.cooldown = self.max_cooldown
        
        # Обновление снарядов
        for projectile in self.projectiles[:]:
            projectile['lifetime'] -= dt
            
            if projectile['lifetime'] <= 0:
                self.projectiles.remove(projectile)
                continue
            
            # Движение снаряда
            projectile['x'] += projectile['dx'] * projectile['speed'] * dt
            projectile['y'] += projectile['dy'] * projectile['speed'] * dt
            
            # Проверка столкновения с врагами
            hit = False
            for enemy in enemies:
                if (projectile['x'] >= enemy.x and
                    projectile['x'] <= enemy.x + enemy.width and
                    projectile['y'] >= enemy.y and
                    projectile['y'] <= enemy.y + enemy.height):
                    
                    if enemy.take_damage(self.damage):
                        # Враг уничтожен, начисление кредитов
                        self.parent.credits += enemy.worth
                        enemies.remove(enemy)
                    
                    hit = True
                    break
            
            if hit:
                self.projectiles.remove(projectile)
        
        self.update_hitbox()
    
    def draw(self, surface):
        # Отрисовка оружия
        rotated_image = pygame.transform.rotate(self.image, self.angle)
        new_rect = rotated_image.get_rect(center=self.image.get_rect(topleft=(self.x, self.y)).center)
        surface.blit(rotated_image, new_rect.topleft)
        
        # Радиус действия (при отладке)
        # pygame.draw.circle(surface, self.color, (self.x + self.width // 2, self.y + self.height // 2), self.range, 1)
        
        # Отрисовка снарядов
        for projectile in self.projectiles:
            pygame.draw.circle(surface, self.color, (int(projectile['x']), int(projectile['y'])), projectile['size'])
    
    def fire(self):
        if not self.target:
            return
        
        # Создание снаряда
        start_x = self.x + self.width // 2
        start_y = self.y + self.height // 2
        
        target_x = self.target.x + self.target.width // 2
        target_y = self.target.y + self.target.height // 2
        
        dx = target_x - start_x
        dy = target_y - start_y
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance > 0:
            dx /= distance
            dy /= distance
        
        projectile = {
            'x': start_x,
            'y': start_y,
            'dx': dx,
            'dy': dy,
            'speed': 10,
            'size': 5,
            'damage': self.damage,
            'lifetime': self.range / 10  # Время жизни снаряда зависит от дальности
        }
        
        self.projectiles.append(projectile)
        
        # Воспроизведение звука выстрела
        if self.weapon_type == 0:  # Laser
            pygame.mixer.Sound('sounds/laser.wav' if os.path.exists('sounds/laser.wav') else None).play()
        elif self.weapon_type == 1:  # Plasma
            pygame.mixer.Sound('sounds/plasma.wav' if os.path.exists('sounds/plasma.wav') else None).play()
        elif self.weapon_type == 2:  # Ion
            pygame.mixer.Sound('sounds/ion.wav' if os.path.exists('sounds/ion.wav') else None).play()
        else:  # Missile
            pygame.mixer.Sound('sounds/missile.wav' if os.path.exists('sounds/missile.wav') else None).play()

class ParticleSystem:
    def __init__(self):
        self.particles = []
    
    def add_explosion(self, x, y, color=YELLOW, num_particles=20):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            size = random.uniform(2, 8)
            lifetime = random.uniform(0.5, 2.0)
            
            particle = {
                'x': x,
                'y': y,
                'dx': math.cos(angle) * speed,
                'dy': math.sin(angle) * speed,
                'size': size,
                'color': color,
                'lifetime': lifetime,
                'max_lifetime': lifetime
            }
            
            self.particles.append(particle)
    
    def update(self, dt):
        for particle in self.particles[:]:
            particle['lifetime'] -= dt
            
            if particle['lifetime'] <= 0:
                self.particles.remove(particle)
                continue
            
            particle['x'] += particle['dx'] * dt
            particle['y'] += particle['dy'] * dt
            
            # Постепенное уменьшение размера
            ratio = particle['lifetime'] / particle['max_lifetime']
            particle['size'] *= 0.99
    
    def draw(self, surface):
        for particle in self.particles:
            ratio = particle['lifetime'] / particle['max_lifetime']
            color = particle['color']
            alpha = int(255 * ratio)
            size = int(particle['size'])
            
            if size <= 0:
                continue
            
            pygame.draw.circle(surface, (*color, alpha), (int(particle['x']), int(particle['y'])), size)

class StarField:
    def __init__(self, width, height, num_stars=200):
        self.width = width
        self.height = height
        self.stars = []
        
        for _ in range(num_stars):
            star = {
                'x': random.randint(0, width),
                'y': random.randint(0, height),
                'size': random.uniform(0.5, 3),
                'brightness': random.uniform(0.5, 1.0),
                'speed': random.uniform(0.1, 0.5)
            }
            self.stars.append(star)
    
    def update(self, dt):
        for star in self.stars:
            # Мерцание звезд
            star['brightness'] += random.uniform(-0.05, 0.05)
            star['brightness'] = max(0.3, min(1.0, star['brightness']))
            
            # Легкое движение звезд (эффект параллакса)
            star['y'] += star['speed'] * dt
            
            # Перенос звезд, вышедших за пределы экрана
            if star['y'] > self.height:
                star['y'] = 0
                star['x'] = random.randint(0, self.width)
    
    def draw(self, surface):
        for star in self.stars:
            brightness = int(255 * star['brightness'])
            color = (brightness, brightness, brightness)
            pygame.draw.circle(surface, color, (int(star['x']), int(star['y'])), star['size'])

class Button:
    def __init__(self, x, y, width, height, text, color=BLUE, hover_color=CYAN, text_color=WHITE):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.is_hovered = False
        self.font = pygame.font.SysFont(None, 24)
    
    def draw(self, surface):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, WHITE, self.rect, 2)  # Border
        
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)
    
    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)
        return self.is_hovered
    
    def is_clicked(self, pos, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(pos)
        return False

class GameState(Enum):
    MENU = 0
    PLAYING = 1
    PAUSED = 2
    GAME_OVER = 3
    SHOP = 4
    VICTORY = 5

class SpaceDefenderGame:
    def __init__(self):
        # Настройка окна
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()
        
        # Состояние игры
        self.state = GameState.MENU
        self.level = 1
        self.wave = 1
        self.score = 0
        self.time_elapsed = 0
        self.wave_timer = 0
        self.enemies_in_wave = 10
        self.enemies_spawned = 0
        self.wave_interval = 30  # Секунды между волнами
        
        # Игровые объекты
        self.station = SpaceStation(SCREEN_WIDTH // 2 - 60, SCREEN_HEIGHT // 2 - 60)
        self.weapons = []
        self.enemies = []
        self.particles = ParticleSystem()
        self.starfield = StarField(SCREEN_WIDTH, SCREEN_HEIGHT)
        
        # Добавление стандартного оружия
        self.weapons.append(Weapon(0, 0, 0, self.station))  # Лазер
        
        # Меню и интерфейс
        self.buttons = {
            GameState.MENU: [
                Button(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 - 50, 200, 50, "Начать игру"),
                Button(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 20, 200, 50, "Выход")
            ],
            GameState.PAUSED: [
                Button(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 - 50, 200, 50, "Продолжить"),
                Button(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 20, 200, 50, "Главное меню")
            ],
            GameState.GAME_OVER: [
                Button(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 50, 200, 50, "Начать заново"),
                Button(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 120, 200, 50, "Главное меню")
            ],
            GameState.VICTORY: [
                Button(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 50, 200, 50, "Следующий уровень"),
                Button(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 120, 200, 50, "Главное меню")
            ],
            GameState.SHOP: [
                Button(SCREEN_WIDTH // 2 - 300, SCREEN_HEIGHT // 2 - 100, 200, 50, "Лазер (100 кр.)"),
                Button(SCREEN_WIDTH // 2 - 300, SCREEN_HEIGHT // 2 - 30, 200, 50, "Плазма (200 кр.)"),
                Button(SCREEN_WIDTH // 2 + 100, SCREEN_HEIGHT // 2 - 100, 200, 50, "Ионная пушка (300 кр.)"),
                Button(SCREEN_WIDTH // 2 + 100, SCREEN_HEIGHT // 2 - 30, 200, 50, "Ракеты (400 кр.)"),
                Button(SCREEN_WIDTH // 2 - 300, SCREEN_HEIGHT // 2 + 40, 200, 50, "Улучшить щиты (150 кр.)"),
                Button(SCREEN_WIDTH // 2 + 100, SCREEN_HEIGHT // 2 + 40, 200, 50, "Улучшить корпус (150 кр.)"),
                Button(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 110, 200, 50, "Вернуться в игру")
            ]
        }
        
        # Загрузка звуков
        self.sounds = {}
        sounds_path = "sounds"
        if not os.path.exists(sounds_path):
            os.makedirs(sounds_path)
            self.create_default_sounds(sounds_path)
        
        self.load_sounds()
        
        # Шрифты
        self.title_font = pygame.font.SysFont(None, 64)
        self.menu_font = pygame.font.SysFont(None, 36)
        self.game_font = pygame.font.SysFont(None, 24)
        self.small_font = pygame.font.SysFont(None, 18)
        
    def create_default_sounds(self, path):
        """Создает базовые звуковые файлы, если они отсутствуют"""
        # Создание простых звуковых сигналов с помощью wave и struct
        try:
            import struct
            
            # Функция для создания простого звукового сигнала
            def create_sound(filename, frequency, duration, volume=0.5):
                sample_rate = 44100
                num_samples = int(sample_rate * duration)
                audio_data = []
                
                for i in range(num_samples):
                    t = float(i) / sample_rate
                    value = int(32767.0 * volume * math.sin(2 * math.pi * frequency * t))
                    audio_data.append(struct.pack('h', value))
                
                audio_data = b''.join(audio_data)
                
                with wave.open(os.path.join(path, filename), 'wb') as wave_file:
                    wave_file.setnchannels(1)
                    wave_file.setsampwidth(2)
                    wave_file.setframerate(sample_rate)
                    wave_file.writeframes(audio_data)
            
            # Создание различных звуков для игры
            create_sound("laser.wav", 880, 0.2, 0.3)  # Высокий короткий звук для лазера
            create_sound("plasma.wav", 220, 0.3, 0.5)  # Низкий звук для плазмы
            create_sound("ion.wav", 440, 0.3, 0.4)  # Средний звук для ионной пушки
            create_sound("missile.wav", 110, 0.5, 0.6)  # Очень низкий звук для ракеты
            create_sound("explosion.wav", 90, 0.8, 0.8)  # Взрыв
            create_sound("hit.wav", 220, 0.1, 0.4)  # Попадание
            create_sound("button.wav", 660, 0.1, 0.2)  # Клик кнопки
            create_sound("victory.wav", [440, 550, 660], 1.0, 0.5)  # Победа
            create_sound("game_over.wav", [440, 330, 220], 1.0, 0.5)  # Поражение
        
        except Exception as e:
            logger.error(f"Ошибка при создании звуковых файлов: {str(e)}")
    
    def load_sounds(self):
        """Загружает звуковые файлы в игру"""
        sounds_path = "sounds"
        for sound_name in ["laser", "plasma", "ion", "missile", "explosion", "hit", "button", "victory", "game_over"]:
            file_path = os.path.join(sounds_path, f"{sound_name}.wav")
            if os.path.exists(file_path):
                try:
                    self.sounds[sound_name] = pygame.mixer.Sound(file_path)
                except Exception as e:
                    logger.error(f"Ошибка при загрузке звука {sound_name}: {str(e)}")
    
    def play_sound(self, sound_name):
        """Воспроизводит звук по имени"""
        if sound_name in self.sounds:
            self.sounds[sound_name].play()
    
    def start_new_game(self):
        """Начинает новую игру"""
        self.state = GameState.PLAYING
        self.level = 1
        self.wave = 1
        self.score = 0
        self.time_elapsed = 0
        self.wave_timer = 0
        self.enemies_in_wave = 10
        self.enemies_spawned = 0
        
        # Сброс космической станции
        self.station = SpaceStation(SCREEN_WIDTH // 2 - 60, SCREEN_HEIGHT // 2 - 60)
        self.weapons = []
        self.enemies = []
        
        # Добавление стандартного оружия
        self.weapons.append(Weapon(0, 0, 0, self.station))  # Лазер
    
    def spawn_enemy(self):
        """Создает нового врага"""
        # Определяем тип врага в зависимости от волны
        enemy_type = min(3, self.wave // 3)
        if random.random() < 0.2:  # 20% шанс появления более сильного врага
            enemy_type = min(3, enemy_type + 1)
        
        # Определяем позицию появления (за пределами экрана)
        side = random.randint(0, 3)  # 0: верх, 1: право, 2: низ, 3: лево
        
        if side == 0:  # Сверху
            x = random.randint(0, SCREEN_WIDTH)
            y = -50
        elif side == 1:  # Справа
            x = SCREEN_WIDTH + 50
            y = random.randint(0, SCREEN_HEIGHT)
        elif side == 2:  # Снизу
            x = random.randint(0, SCREEN_WIDTH)
            y = SCREEN_HEIGHT + 50
        else:  # Слева
            x = -50
            y = random.randint(0, SCREEN_HEIGHT)
        
        # Создаем и добавляем врага
        enemy = Enemy(x, y, enemy_type)
        self.enemies.append(enemy)
        self.enemies_spawned += 1
    
    def update(self, dt):
        """Обновляет состояние игры"""
        # Обновляем звездное поле
        self.starfield.update(dt)
        
        if self.state == GameState.PLAYING:
            # Обновляем время
            self.time_elapsed += dt
            self.wave_timer += dt
            
            # Проверяем, нужно ли начать новую волну
            if self.enemies_spawned < self.enemies_in_wave and len(self.enemies) < 10:
                if random.random() < 0.02:  # 2% шанс появления врага каждый кадр
                    self.spawn_enemy()
            
            # Если все враги в волне уничтожены
            if self.enemies_spawned >= self.enemies_in_wave and len(self.enemies) == 0:
                if self.wave % 5 == 0:  # Каждые 5 волн - магазин
                    self.state = GameState.SHOP
                else:
                    self.wave += 1
                    self.enemies_in_wave = 10 + (self.wave - 1) * 5  # Увеличиваем число врагов
                    self.enemies_spawned = 0
                    self.wave_timer = 0
            
            # Обновляем станцию
            self.station.update(dt)
            
            # Обновляем оружие
            for weapon in self.weapons:
                weapon.update(dt, self.enemies)
            
            # Обновляем врагов
            for enemy in self.enemies[:]:
                if enemy.update(dt, self.station):
                    # Враг достиг станции и атакует
                    if enemy.attack_cooldown <= 0:
                        if self.station.take_damage(enemy.damage):
                            # Станция уничтожена
                            self.state = GameState.GAME_OVER
                            self.play_sound("game_over")
                        
                        enemy.attack_cooldown = 1.0  # 1 секунда между атаками
                        self.play_sound("hit")
            
            # Обновляем частицы
            self.particles.update(dt)
    
    def handle_shop(self, pos, event):
        """Обрабатывает действия в магазине"""
        buttons = self.buttons[GameState.SHOP]
        
        for i, button in enumerate(buttons):
            if button.is_clicked(pos, event):
                self.play_sound("button")
                
                if i == 0:  # Лазер
                    if self.station.credits >= 100:
                        self.station.credits -= 100
                        self.weapons.append(Weapon(0, 0, 0, self.station))
                
                elif i == 1:  # Плазма
                    if self.station.credits >= 200:
                        self.station.credits -= 200
                        self.weapons.append(Weapon(0, 0, 1, self.station))
                
                elif i == 2:  # Ионная пушка
                    if self.station.credits >= 300:
                        self.station.credits -= 300
                        self.weapons.append(Weapon(0, 0, 2, self.station))
                
                elif i == 3:  # Ракеты
                    if self.station.credits >= 400:
                        self.station.credits -= 400
                        self.weapons.append(Weapon(0, 0, 3, self.station))
                
                elif i == 4:  # Улучшить щиты
                    if self.station.credits >= 150:
                        self.station.credits -= 150
                        self.station.max_shield += 100
                        self.station.shield += 100
                
                elif i == 5:  # Улучшить корпус
                    if self.station.credits >= 150:
                        self.station.credits -= 150
                        self.station.max_health += 100
                        self.station.health += 100
                
                elif i == 6:  # Вернуться в игру
                    self.state = GameState.PLAYING
                    self.wave += 1
                    self.enemies_in_wave = 10 + (self.wave - 1) * 5
                    self.enemies_spawned = 0
                    self.wave_timer = 0
    
    def handle_events(self):
        """Обрабатывает ввод пользователя"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.state == GameState.PLAYING:
                        self.state = GameState.PAUSED
                    elif self.state == GameState.PAUSED:
                        self.state = GameState.PLAYING
            
            # Обработка кликов мыши
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                
                if self.state == GameState.MENU:
                    for i, button in enumerate(self.buttons[GameState.MENU]):
                        if button.is_clicked(pos, event):
                            self.play_sound("button")
                            if i == 0:  # Начать игру
                                self.start_new_game()
                            elif i == 1:  # Выход
                                return False
                
                elif self.state == GameState.PAUSED:
                    for i, button in enumerate(self.buttons[GameState.PAUSED]):
                        if button.is_clicked(pos, event):
                            self.play_sound("button")
                            if i == 0:  # Продолжить
                                self.state = GameState.PLAYING
                            elif i == 1:  # Главное меню
                                self.state = GameState.MENU
                
                elif self.state == GameState.GAME_OVER:
                    for i, button in enumerate(self.buttons[GameState.GAME_OVER]):
                        if button.is_clicked(pos, event):
                            self.play_sound("button")
                            if i == 0:  # Начать заново
                                self.start_new_game()
                            elif i == 1:  # Главное меню
                                self.state = GameState.MENU
                
                elif self.state == GameState.VICTORY:
                    for i, button in enumerate(self.buttons[GameState.VICTORY]):
                        if button.is_clicked(pos, event):
                            self.play_sound("button")
                            if i == 0:  # Следующий уровень
                                self.level += 1
                                self.state = GameState.PLAYING
                            elif i == 1:  # Главное меню
                                self.state = GameState.MENU
                
                elif self.state == GameState.SHOP:
                    self.handle_shop(pos, event)
        
        # Обновление состояния наведения кнопок
        pos = pygame.mouse.get_pos()
        if self.state in self.buttons:
            for button in self.buttons[self.state]:
                button.check_hover(pos)
        
        return True
    
    def draw_menu(self):
        """Отрисовка главного меню"""
        # Фон звездного поля
        self.starfield.draw(self.screen)
        
        # Заголовок
        title_text = self.title_font.render("КОСМИЧЕСКИЙ ЗАЩИТНИК", True, CYAN)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4))
        self.screen.blit(title_text, title_rect)
        
        # Кнопки
        for button in self.buttons[GameState.MENU]:
            button.draw(self.screen)
    
    def draw_pause(self):
        """Отрисовка меню паузы"""
        # Полупрозрачный фон
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))
        
        # Заголовок
        title_text = self.title_font.render("ПАУЗА", True, WHITE)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4))
        self.screen.blit(title_text, title_rect)
        
        # Кнопки
        for button in self.buttons[GameState.PAUSED]:
            button.draw(self.screen)
    
    def draw_game_over(self):
        """Отрисовка экрана поражения"""
        # Полупрозрачный фон
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 192))
        self.screen.blit(overlay, (0, 0))
        
        # Заголовок
        title_text = self.title_font.render("ИГРА ОКОНЧЕНА", True, RED)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4))
        self.screen.blit(title_text, title_rect)
        
        # Статистика
        stats_text = self.menu_font.render(f"Уровень: {self.level} | Волна: {self.wave} | Счет: {self.score}", True, WHITE)
        stats_rect = stats_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3))
        self.screen.blit(stats_text, stats_rect)
        
        # Кнопки
        for button in self.buttons[GameState.GAME_OVER]:
            button.draw(self.screen)
    
    def draw_victory(self):
        """Отрисовка экрана победы"""
        # Полупрозрачный фон
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 192))
        self.screen.blit(overlay, (0, 0))
        
        # Заголовок
        title_text = self.title_font.render("ПОБЕДА!", True, GREEN)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4))
        self.screen.blit(title_text, title_rect)
        
        # Статистика
        stats_text = self.menu_font.render(f"Уровень завершен: {self.level} | Счет: {self.score}", True, WHITE)
        stats_rect = stats_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3))
        self.screen.blit(stats_text, stats_rect)
        
        # Кнопки
        for button in self.buttons[GameState.VICTORY]:
            button.draw(self.screen)
    
    def draw_shop(self):
        """Отрисовка магазина"""
        # Полупрозрачный фон
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 192))
        self.screen.blit(overlay, (0, 0))
        
        # Заголовок
        title_text = self.title_font.render("МАГАЗИН", True, YELLOW)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 8))
        self.screen.blit(title_text, title_rect)
        
        # Кредиты
        credits_text = self.menu_font.render(f"Кредиты: {self.station.credits}", True, GREEN)
        credits_rect = credits_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4))
        self.screen.blit(credits_text, credits_rect)
        
        # Кнопки
        for button in self.buttons[GameState.SHOP]:
            button.draw(self.screen)
        
        # Информация о текущем вооружении
        weapons_text = self.game_font.render("Текущее вооружение:", True, WHITE)
        weapons_rect = weapons_text.get_rect(topright=(SCREEN_WIDTH - 20, SCREEN_HEIGHT // 2 + 100))
        self.screen.blit(weapons_text, weapons_rect)
        
        weapon_counts = {'Лазер': 0, 'Плазма': 0, 'Ионная пушка': 0, 'Ракеты': 0}
        for weapon in self.weapons:
            if weapon.weapon_type == 0:
                weapon_counts['Лазер'] += 1
            elif weapon.weapon_type == 1:
                weapon_counts['Плазма'] += 1
            elif weapon.weapon_type == 2:
                weapon_counts['Ионная пушка'] += 1
            elif weapon.weapon_type == 3:
                weapon_counts['Ракеты'] += 1
        
        y_offset = SCREEN_HEIGHT // 2 + 130
        for weapon_name, count in weapon_counts.items():
            if count > 0:
                weapon_info = self.small_font.render(f"{weapon_name}: {count}", True, WHITE)
                self.screen.blit(weapon_info, (SCREEN_WIDTH - 150, y_offset))
                y_offset += 20
    
    def draw_playing(self):
        """Отрисовка игрового процесса"""
        # Фон звездного поля
        self.starfield.draw(self.screen)
        
        # Отрисовка частиц
        self.particles.draw(self.screen)
        
        # Отрисовка врагов
        for enemy in self.enemies:
            enemy.draw(self.screen)
        
        # Отрисовка станции
        self.station.draw(self.screen)
        
        # Отрисовка оружия
        for weapon in self.weapons:
            weapon.draw(self.screen)
        
        # Интерфейс
        # Верхняя панель
        pygame.draw.rect(self.screen, (30, 30, 50), (0, 0, SCREEN_WIDTH, 40))
        pygame.draw.line(self.screen, WHITE, (0, 40), (SCREEN_WIDTH, 40))
        
        # Информация о волне и уровне
        level_text = self.game_font.render(f"Уровень: {self.level}", True, WHITE)
        self.screen.blit(level_text, (20, 10))
        
        wave_text = self.game_font.render(f"Волна: {self.wave}", True, WHITE)
        self.screen.blit(wave_text, (150, 10))
        
        enemies_text = self.game_font.render(f"Враги: {len(self.enemies)}/{self.enemies_in_wave - self.enemies_spawned + len(self.enemies)}", True, WHITE)
        self.screen.blit(enemies_text, (280, 10))
        
        # Информация о станции
        health_text = self.game_font.render(f"Корпус: {self.station.health}/{self.station.max_health}", True, GREEN)
        self.screen.blit(health_text, (450, 10))
        
        shield_text = self.game_font.render(f"Щиты: {int(self.station.shield)}/{self.station.max_shield}", True, CYAN)
        self.screen.blit(shield_text, (650, 10))
        
        # Кредиты
        credits_text = self.game_font.render(f"Кредиты: {self.station.credits}", True, YELLOW)
        self.screen.blit(credits_text, (850, 10))
        
        # Счет
        score_text = self.game_font.render(f"Счет: {self.score}", True, WHITE)
        score_rect = score_text.get_rect(right=SCREEN_WIDTH - 20)
        score_rect.y = 10
        self.screen.blit(score_text, score_rect)
    
    def draw(self):
        """Отрисовка игры в зависимости от состояния"""
        self.screen.fill(BLACK)
        
        if self.state == GameState.MENU:
            self.draw_menu()
        elif self.state == GameState.PLAYING:
            self.draw_playing()
        elif self.state == GameState.PAUSED:
            self.draw_playing()
            self.draw_pause()
        elif self.state == GameState.GAME_OVER:
            self.draw_playing()
            self.draw_game_over()
        elif self.state == GameState.VICTORY:
            self.draw_playing()
            self.draw_victory()
        elif self.state == GameState.SHOP:
            self.draw_playing()
            self.draw_shop()
    
    def run(self):
        """Основной игровой цикл"""
        running = True
        last_time = time.time()
        
        while running:
            # Расчет времени кадра
            current_time = time.time()
            dt = (current_time - last_time) * 1000  # в миллисекундах
            dt = min(dt, 50)  # Ограничение максимального времени кадра
            dt /= 1000  # обратно в секунды
            last_time = current_time
            
            # Обработка событий
            running = self.handle_events()
            
            # Обновление игры
            self.update(dt)
            
            # Отрисовка
            self.draw()
            
            # Обновление экрана
            pygame.display.flip()
            
            # Ограничение FPS
            self.clock.tick(FPS)
        
        # Сохранение прогресса при выходе
        self.save_progress()
        
        pygame.quit()
    
    def save_progress(self):
        """Сохраняет прогресс игры в базу данных"""
        try:
            session = Session()
            
            # Проверяем, есть ли уже запись для игрока
            player = session.query(PlayerData).filter_by(name="player").first()
            
            if not player:
                # Создаем новую запись
                player = PlayerData(
                    name="player",
                    level=self.level,
                    experience=self.score,
                    credits=self.station.credits,
                    station_health=self.station.health
                )
                session.add(player)
            else:
                # Обновляем существующую запись
                player.level = self.level
                player.experience = self.score
                player.credits = self.station.credits
                player.station_health = self.station.health
            
            # Удаляем старые улучшения и оружие
            session.query(UpgradeData).filter_by(player_id=player.id).delete()
            session.query(WeaponData).filter_by(player_id=player.id).delete()
            
            # Сохраняем текущие улучшения и оружие
            for weapon in self.weapons:
                weapon_data = WeaponData(
                    player_id=player.id,
                    name=f"Weapon Type {weapon.weapon_type}",
                    damage=weapon.damage,
                    cooldown=weapon.max_cooldown
                )
                session.add(weapon_data)
            
            # Фиксируем изменения
            session.commit()
            logger.info("Прогресс успешно сохранен")
        
        except Exception as e:
            logger.error(f"Ошибка при сохранении прогресса: {str(e)}")
        finally:
            session.close()
    
    def load_progress(self):
        """Загружает прогресс игры из базы данных"""
        try:
            session = Session()
            
            # Загружаем данные игрока
            player = session.query(PlayerData).filter_by(name="player").first()
            
            if player:
                self.level = player.level
                self.score = player.experience
                self.station.credits = player.credits
                self.station.health = player.station_health
                
                # Загружаем оружие
                weapons_data = session.query(WeaponData).filter_by(player_id=player.id).all()
                
                for weapon_data in weapons_data:
                    # Определяем тип оружия по имени
                    weapon_type = int(weapon_data.name.split()[-1])
                    
                    # Добавляем оружие
                    weapon = Weapon(0, 0, weapon_type, self.station)
                    weapon.damage = weapon_data.damage
                    weapon.max_cooldown = weapon_data.cooldown
                    
                    self.weapons.append(weapon)
                
                logger.info("Прогресс успешно загружен")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке прогресса: {str(e)}")
        finally:
            session.close()

# Запуск игры
if __name__ == "__main__":
    # Отображение ASCII-арта при запуске
    title_art = pyfiglet.figlet_format("Space Defender", font="big")
    for line in title_art.split("\n"):
        print(colorama.Fore.CYAN + line)
    print(colorama.Style.RESET_ALL)
    
    # Создание и запуск игры
    game = SpaceDefenderGame()
    game.run()
