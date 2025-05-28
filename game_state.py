from player import Player
import csv
import os

class GameState:

    def __init__(self, input_dict):

        self.dict_to_object(input_dict)

    def dict_to_object(self, input_dict):
        list1 = [input_dict['frame'], input_dict['timer'], input_dict['result'], input_dict['round_started'], input_dict['round_over'], input_dict['height_delta'], input_dict['width_delta'], 
                input_dict['p2']['character'], input_dict['p2']['health'], input_dict['p2']['x'], input_dict['p2']['y'], input_dict['p2']['jumping'], input_dict['p2']['crouching'], 
                input_dict['p2']['buttons']['Up'], input_dict['p2']['buttons']['Down'], input_dict['p2']['buttons']['Left'], input_dict['p2']['buttons']['Right'],
                input_dict['p2']['buttons']['Select'], input_dict['p2']['buttons']['Start'], input_dict['p2']['buttons']['Y'], input_dict['p2']['buttons']['B'], input_dict['p2']['buttons']['X'], input_dict['p2']['buttons']['A'], input_dict['p2']['buttons']['L'], input_dict['p2']['buttons']['R'], input_dict['p2']['in_move'], input_dict['p2']['move'],
                input_dict['p1']['character'], input_dict['p1']['health'], input_dict['p1']['x'], input_dict['p1']['y'], input_dict['p1']['jumping'], input_dict['p1']['crouching'], 
                input_dict['p1']['in_move'], input_dict['p1']['move'],
                input_dict['p1']['buttons']['Up'], input_dict['p1']['buttons']['Down'], input_dict['p1']['buttons']['Left'], input_dict['p1']['buttons']['Right'],
                input_dict['p1']['buttons']['Select'], input_dict['p1']['buttons']['Start'], input_dict['p1']['buttons']['Y'], input_dict['p1']['buttons']['B'], input_dict['p1']['buttons']['X'], input_dict['p1']['buttons']['A'], input_dict['p1']['buttons']['L'], input_dict['p1']['buttons']['R']]

        if list1[0] % 10 == 0:
            print("Before writing to file:")
            print(list1)

            header = [
                'frame', 'timer', 'result', 'round_started', 'round_over', 'height_delta', 'width_delta',
                'player2_character', 'player2_health', 'player2_x', 'player2_y', 'player2_jumping', 'player2_crouching',
                'player2_Up', 'player2_Down', 'player2_Left', 'player2_Right',
                'player2_Select', 'player2_Start', 'player2_Y', 'player2_B', 'player2_X', 'player2_A', 'player2_L', 'player2_R',
                'player2_in_move', 'player2_move',
                'player1_character', 'player1_health', 'player1_x', 'player1_y', 'player1_jumping', 'player1_crouching',
                'player1_in_move', 'player1_move',
                'player1_Up', 'player1_Down', 'player1_Left', 'player1_Right',
                'player1_Select', 'player1_Start', 'player1_Y', 'player1_B', 'player1_X', 'player1_A', 'player1_L', 'player1_R'
            ]

            file_name = 'dataset.csv'

            try:
                check = os.path.exists(file_name)
                with open(file_name, 'a', newline='') as file:
                    writer = csv.writer(file)
                    if not check:
                        writer.writerow(header)
                    writer.writerow(list1)
            except Exception as e:
                print(f"Error while writing to {file_name}: {e}")
        
        self.player1 = Player(input_dict['p1'])
        self.player2 = Player(input_dict['p2'])
        self.timer = input_dict['timer']
        self.fight_result = input_dict['result']
        self.has_round_started = input_dict['round_started']
        self.is_round_over = input_dict['round_over']