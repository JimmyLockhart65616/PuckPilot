"""In-season team management: daily lineups, weekly category plans, waivers.

The draft package reasons about one evening. This one reasons about 185 game
days, which changes two things structurally.

First, the date axis. Every engine under `engine/` and `draft/` is driven by an
integer index into `ReplayData.dates`, and those dates come from game logs -
games that have already been played. Nothing there can answer "who plays
tomorrow". `season.calendar` builds the forward axis from `nhl_schedule`.

Second, whose team it is. The draft board learned late that `my_seat` belongs in
a parameter rather than in the object; here that lesson is the starting point.
Nothing in this package holds a global "my team": the manager is passed in.
"""
