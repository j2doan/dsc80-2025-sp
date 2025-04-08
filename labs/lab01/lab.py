# lab.py


from pathlib import Path
import io
import pandas as pd
import numpy as np
np.set_printoptions(legacy='1.21')


# ---------------------------------------------------------------------
# QUESTION 0
# ---------------------------------------------------------------------


def consecutive_ints(ints):
    if len(ints) == 0:
        return False

    for k in range(len(ints) - 1):
        diff = abs(ints[k] - ints[k+1])
        if diff == 1:
            return True

    return False


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def median_vs_mean(nums):
    sorted_nums = sorted(nums)
    length = len(nums)
    possible_odd = length - 1

    median = (sorted_nums[length//2] + sorted_nums[possible_odd//2]) / 2
    mean = sum(sorted_nums) / length
    
    return median <= mean


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def n_prefixes(s, n):
    new_str = ""
    for i in range(n, 0, -1):
        new_str += s[:i]
    return new_str


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def exploded_numbers(ints, n):
    final_list = []
    for int in ints:
        curr_int = int

        max_str_len = 0
        for j in ints:
            if max_str_len <  len(str(j + n)):
                max_str_len = len(str(j + n))

        curr_list = [str(int).zfill(max_str_len)]

        for j in range(1, n + 1):
            curr_list.insert(0, str(curr_int - j).zfill(max_str_len))
            curr_list.append(str(int + j).zfill(max_str_len))
        

        final_list.append(" ".join(curr_list))

    return final_list




# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def last_chars(fh):
    output = ""
    for line in fh:
        for j in range(-1, (len(line) * -1) -1, -1):
            if line[j] != '\n':
                break
        output = output + line[j]
    return output
    


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def add_root(A):
    root_list = np.arange(len(A))
    A = A + np.sqrt(root_list)
    return A


def where_square(A):
    return np.floor(np.sqrt(A)) ** 2 == A


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def filter_cutoff_loop(matrix, cutoff):
    output = [[] for _ in range(len(matrix))]

    for col in range(len(matrix[0])):

        total = []
        for row in range(len(matrix)):
            total.append(matrix[row][col])

        if sum(total) / len(matrix) > cutoff:
            for i in range(len(output)):
                output[i].append(total[i])
                
    return np.array(output)



# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def filter_cutoff_np(matrix, cutoff):
    means = np.mean(matrix, axis=0)
    check = means > cutoff
    return matrix[:, check]

# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def growth_rates(A):
    growth = (A[1:] - A[:-1]) / A[:-1]
    return np.round(growth, 2)

def with_leftover(A):
    leftovers = 20 % A
    cumulative_leftovers = np.cumsum(leftovers)
    can_buy = cumulative_leftovers >= A
    return len(A) - np.sum(can_buy)


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def salary_stats(salary):
    num_players = salary['Player'].nunique() # player count
    num_teams = salary['Team'].nunique() # team count
    total_salary = salary['Salary'].sum() # total salary
    highest_salary = salary.sort_values(by='Salary', ascending=False)['Player'].iloc[0] # highest salary
    avg_loss = salary.groupby('Team')['Salary'].mean().loc['Los Angeles Lakers'] # avg loss LA Lakers
    fifth_lowest_salary = salary.sort_values(by='Salary')[['Player', 'Team']].iloc[4]
    fifth_lowest = fifth_lowest_salary.loc['Player'] + ', ' + fifth_lowest_salary.loc['Team']
    stripped_suffix = salary['Player'].str.replace(' Jr.', '').str.replace(' III', '') 
    duplicates = stripped_suffix.str.split(' ').apply(lambda x: x[1]).nunique() != stripped_suffix.nunique()
    team_of_highest_paid = salary.sort_values(by='Salary', ascending=False)['Team'].iloc[0]
    total_highest = salary.groupby('Team')['Salary'].sum().loc[team_of_highest_paid]

    index = ['num_players', 'num_teams', 'total_salary', 'highest_salary', 'avg_loss', 'fifth_lowest', 'duplicates', 'total_highest']
    data = [num_players, num_teams, total_salary, highest_salary, avg_loss, fifth_lowest, duplicates, total_highest]
    return pd.Series(data, index=index)


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def parse_malformed(fp):
    with open(fp) as fs:
        data = fs.readlines()

    new_lines = []
    for line in data[1:]:
        
        line = line.strip() # unecessary whitespace
        
        line = line.replace('"', '') # no need for quotations
        
        line = line.replace(',,', ',') # get rid of ,,
        
        if line[0] == ',': # get rid of commas at either end
            line = line[1:]
        if line[-1] == ',':
            line = line[:-1]

        last_comma_index = line.rfind(',') # convert , to _ so when split, the comma stays
        if last_comma_index != -1:
            line = line[:last_comma_index] + "_" + line[last_comma_index + 1:]
        col = line.split(',')
        col[-1] = col[-1].replace('_', ',')

        new_lines.append(col) # append

    new_lines.insert(0, data[0].strip().split(','))

    df = pd.DataFrame(new_lines[1:], columns=new_lines[0])

    df['weight'] = pd.to_numeric(df['weight']).astype('float64')
    df['height'] = pd.to_numeric(df['height']).astype('float64')
    
    return df
