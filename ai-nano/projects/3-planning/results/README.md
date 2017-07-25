
_Udacity Artificial Intelligence Nanodegree, July 2017_
# Project 3: Implement a Planning Search
The **goal** of this project is to build a planning search agent that finds the optimal shipping routes for an air cargo transport system. 

The project includes basic skeletons for the classes and functions needed, but students must complete the missing pieces described below.

---
## Part 1: Planning Problems

#### 1. Implement methods and functions in `my_air_cargo_problems.py`

**1.a.** `AirCargoProblem.get_actions` method including `load_actions` and `unload_actions` sub-functions

_Create concrete Action objects based on the domain action schema for: Load, Unload, and Fly. A concrete action is a specific literal action that does not include variables as with the schema. For example, the action schema 'Load(c, p, a)' can represent the concrete actions 'Load(C1, P1, SFO)' or 'Load(C2, P2, JFK)'. The actions for the planning problem must be concrete because the problems in forward search and planning graphs must use propositional logic._

The source code for my solution can be found here. Below is a snippet showing the implementation of the Load action.

```
def load_actions():
            """Create all concrete Load actions and return a list

            :return: list of Action objects
            """
            loads = []
            for a in self.airports:
                for p in self.planes:
                    for c in self.cargos:
                        # preconditions - make sure cargo and plane are At airport
                        precond_pos = [
                            expr("At({}, {})".format(c, a)),
                            expr("At({}, {})".format(p, a)),
                        ]
                        precond_neg = []
                        # positive action - put cargo In plane
                        effect_add = [expr("In({}, {})".format(c, p))]
                        # negative action - remove cargo At airport
                        effect_rem = [expr("At({}, {})".format(c, a))]
                        load = Action(expr("Load({}, {}, {})".format(c, p, a)),
                                        [precond_pos, precond_neg],
                                        [effect_add, effect_rem])
                        loads.append(load)
            return loads
```

**1.b** `AirCargoProblem.actions` method

**1.c** `AirCargoProblem.result` method

**1.d** `air_cargo_p2` function

**1.e** `air_cargo_p3` function


#### 2. Experiment and document metrics for non-heuristic planning solution searches
_Run uninformed planning searches for `air_cargo_p1`, `air_cargo_p2`, and `air_cargo_p3`. Provide metrics on number of node expansions required, number of goal tests, time elapsed, and optimality of solution for each search algorithm. Include the result of at least three of these searches, including `breadth_first_search` and `depth_first_graph_search` in your write-up.



---
## Part 2: Domain-independent heuristics

#### 3. Implement heuristic method in `my_air_cargo_problems.py`

`AirCargoProblem.h_ignore_preconditions` method

#### 4. Implement a Planning Graph with automatic heuristics in `my_planning_graph.py`

`PlanningGraph.add_action_level` method
`PlanningGraph.add_literal_level` method
`PlanningGraph.inconsistent_effects_mutex` method
`PlanningGraph.interference_mutex` method
`PlanningGraph.competing_needs_mutex` method
`PlanningGraph.negation_mutex` method
`PlanningGraph.inconsistent_support_mutex` method
`PlanningGraph.h_levelsum` method

#### 5. Experiment and document metrics of A* searches with these heuristics
_Run A* planning searches using the heuristics you have implemented on `air_cargo_p1`, `air_cargo_p2` and `air_cargo_p3`. Provide metrics on number of node expansions required, number of goal tests, time elapsed, and optimality of solution for each search algorithm and include the results in your report._


---
## Part 3: Written Analysis

![problem 1](problem-1.jpg)


![problem 2](problem-2.jpg)


![problem 3](problem-3.jpg)




---
## Part 4: Research Review
### Instructions
The field of Artificial lIntelligence is continually changing and advancing. To be an AI Engineer at the cutting edge of your field, you need to be able to read and communicate some of these advancements with your peers. In order to help you get comfortable with this, in the second part of this project you will read a seminal paper in the field of Game-Playing and write a simple one page summary on it.

Write a simple one page summary covering the paper's goals, the techniques introduced, and results (if any).

### My Research Review
[Here is a link](https://github.com/tommytracey/udacity/tree/master/ai-nano/projects/2-isolation/results/research_review.pdf) to a PDF version of my research review on AlphaGo. The paper is titled, [Mastering the Game of Go with Deep Neural Networks and Tree Search](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf), written by the team at Deep Mind and featured in the journal [Nature](https://www.nature.com/nature/journal/v529/n7587/full/nature16961.html) in January, 2016.

---
