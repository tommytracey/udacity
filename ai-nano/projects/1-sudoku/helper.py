rows = 'ABCDEFGHI'
cols = '123456789'

def cross(a, b):
	return [s+t for s in a for t in b]

boxes = cross(rows, cols)

# print(boxes)

row_units = [cross(r, cols) for r in rows]

# print(row_units[0])

col_units = [cross(rows, c) for c in cols]

# print(col_units[0])

square_units = [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')]

# print(square_units[0])

unitlist = row_units + column_units + square_units