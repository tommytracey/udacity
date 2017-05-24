rows = 'ABCDEFGHI'
cols = '123456789'

def cross(a, b):
	return [s+t for s in a for t in b]

boxes = cross(rows, cols)

# print(boxes)

row_units = [cross(r, cols) for r in rows]

# print(row_units)

col_units = [cross(rows, c) for c in cols]

# print(col_units[0])

square_units = [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')]

# print(square_units[0])

def diagonal(rows):
	col_a, col_b = 1, 9
	diag_a, diag_b = [], []
	for r in rows:
		box_a, box_b = r + str(col_a), r + str(col_b)
		diag_a.append(box_a)
		diag_b.append(box_b)
		col_a += 1
		col_b -= 1
	return [diag_a, diag_b]

diag_units = diagonal(rows)

# print(diag_units)

unitlist = row_units + col_units + square_units + diag_units