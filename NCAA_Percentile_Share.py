import matplotlib.pyplot as plt
import pandas as pd

import os
from natsort import natsorted
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from streamlit.hashing import _CodeHasher
import matplotlib.font_manager as fm
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cbook as cbook
import matplotlib.image as image
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections import get_projection_class
from scipy.spatial import ConvexHull
import vaex




try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server


@st.cache(allow_output_mutation=True)
def load_data():
    #url = 'https://drive.google.com/file/d/1KD5nxMlZZImiArxLXg4N43uiUkJ4taQP/view?usp=sharing'
    #path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    path = './NCAA_2021.csv'
    df = pd.read_csv(path)
    #df = df[(df.Team.isin(player)) & (df.Season.isin(season))]
    return df


def main():
    state = _get_state()
    pages = {
        "Forward Percentile": FWPercentile,
        "Attacking Midfield Percentile": AMPercentile,
        "Defenisve Midfield Percentile": DMPercentile,
        "Defense Percentile": DPercentile,
    }

    # st.sidebar.title("Page Filters")
    page = st.sidebar.radio("Select Page", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()


def display_state_values(state):
    st.write("Input state:", state.input)
    st.write("Slider state:", state.slider)
    # st.write("Radio state:", state.radio)
    st.write("Checkbox state:", state.checkbox)
    st.write("Selectbox state:", state.selectbox)
    st.write("Multiselect state:", state.multiselect)

    for i in range(3):
        st.write(f"Value {i}:", state[f"State value {i}"])

    if st.button("Clear state"):
        state.clear()


def multiselect(label, options, default, format_func=str):
    """multiselect extension that enables default to be a subset list of the list of objects
     - not a list of strings

     Assumes that options have unique format_func representations

     cf. https://github.com/streamlit/streamlit/issues/352
     """
    options_ = {format_func(option): option for option in options}
    default_ = [format_func(option) for option in default]
    selections = st.multiselect(
        label, options=list(options_.keys()), default=default_, format_func=format_func
    )
    return [options_[format_func(selection)] for selection in selections]


# selections = multiselect("Select", options=[Option1, Option2], default=[Option2])


class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()

    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False

        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")

    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


def FWPercentile(state):
    df = load_data()

    df['Per90'] = df.Minutes / 90
    df['NPenGoals'] = df.Goals - df.Penalties
    df['TackInt'] = df['Successful Tackles'] + df['Interceptions']
    df['NPGoals/90'] = df.NPenGoals / df.Per90
    df['xG/90'] = df.xG / df.Per90
    df['xA/90'] = df.xA / df.Per90
    df['Assist/90'] = df.Assists / df.Per90
    df['Shots/90'] = df.Shots / df.Per90
    df['Passes/90'] = df.Passes / df.Per90
    df['PassesComplete/90'] = df['Passes Complete'] / df.Per90
    df['KeyPasses/90'] = df['Key Passes'] / df.Per90
    df['CrossesComplete/90'] = df['Crosses Complete'] / df.Per90
    df['D2LostBalls/90'] = df['D2 Lost Balls'] / df.Per90
    df['AerialsWon/90'] = df['Aerials Won'] / df.Per90
    df['DribblesSuccessful/90'] = df['Dribbles Successful'] / df.Per90
    df['TackInt/90'] = df['TackInt'] / df.Per90
    df['TacklesWon/90'] = df['Successful Tackles'] / df.Per90
    df['Interceptions/90'] = df.Interceptions / df.Per90
    df['A2Recoveries/90'] = df['A2 Recoveries'] / df.Per90
    df['Recoveries/90'] = df['Ball Recoveries'] / df.Per90

    totaldf = pd.DataFrame(data=df,
                      columns=['Team', 'Player', 'Position', 'Nationality', 'NPGoals/90', 'xG/90', 'Assist/90', 'xA/90',
                               'Shots/90',
                               'Shot%', 'Passes/90', 'PassesComplete/90', 'Pass%', 'KeyPasses/90', 'CrossesComplete/90',
                               'DribblesSuccessful/90', 'Dribble%', 'D2LostBalls/90', 'AerialsWon/90', 'Aerial%', 'Tackle%',
                               'TacklesWon/90',
                               'Interceptions/90', 'TackInt/90', 'A2Recoveries/90', 'Recoveries/90'])

    df = pd.DataFrame(data=df,
                      columns=['Team', 'Player', 'Position', 'Nationality', 'NPGoals/90', 'xG/90', 'Assist/90', 'xA/90',
                               'Shots/90',
                               'Shot%', 'Passes/90', 'PassesComplete/90', 'Pass%', 'KeyPasses/90', 'CrossesComplete/90',
                               'DribblesSuccessful/90', 'Dribble%', 'D2LostBalls/90', 'AerialsWon/90', 'Aerial%', 'Tackle%',
                               'TacklesWon/90',
                               'Interceptions/90', 'TackInt/90', 'A2Recoveries/90', 'Recoveries/90'])

    df['NPGoals/90'] = df['NPGoals/90'].rank(pct=True)
    df['xG/90'] = df['xG/90'].rank(pct=True)
    df['Assist/90'] = df['Assist/90'].rank(pct=True)
    df['xA/90'] = df['xA/90'].rank(pct=True)
    df['Shots/90'] = df['Shots/90'].rank(pct=True)
    df['Shot%'] = df['Shot%'].rank(pct=True)
    df['Passes/90'] = df['Passes/90'].rank(pct=True)
    df['PassesComplete/90'] = df['PassesComplete/90'].rank(pct=True)
    df['Pass%'] = df['Pass%'].rank(pct=True)
    df['KeyPasses/90'] = df['KeyPasses/90'].rank(pct=True)
    df['CrossesComplete/90'] = df['CrossesComplete/90'].rank(pct=True)
    df['DribblesSuccessful/90'] = df['DribblesSuccessful/90'].rank(pct=True)
    df['D2LostBalls/90'] = df['D2LostBalls/90'].rank(pct=True)
    df['AerialsWon/90'] = df['AerialsWon/90'].rank(pct=True)
    df['Aerial%'] = df['Aerial%'].rank(pct=True)
    df['Tackle%'] = df['Tackle%'].rank(pct=True)
    df['Dribble%'] = df['Dribble%'].rank(pct=True)
    df['TacklesWon/90'] = df['TacklesWon/90'].rank(pct=True)
    df['Interceptions/90'] = df['Interceptions/90'].rank(pct=True)
    df['TackInt/90'] = df['TackInt/90'].rank(pct=True)
    df['A2Recoveries/90'] = df['A2Recoveries/90'].rank(pct=True)
    df['Recoveries/90'] = df['Recoveries/90'].rank(pct=True)


    player = st.sidebar.multiselect('Select Player', natsorted(df.Player.unique()))

    FW = pd.DataFrame(data=df,
                      columns=['Team', 'Player', 'Position', 'Nationality', 'NPGoals/90', 'xG/90', 'Shots/90', 'Shot%',
                               'Assist/90', 'xA/90', 'KeyPasses/90',
                               'CrossesComplete/90', 'Pass%', 'DribblesSuccessful/90', 'A2Recoveries/90', 'Aerial%'])
    AM = pd.DataFrame(data=df, columns=['Team', 'Player', 'Position', 'Nationality', 'NPGoals/90', 'xG/90', 'Shots/90',
                                       'Assist/90', 'xA/90', 'Passes/90', 'KeyPasses/90',
                                       'Pass%', 'DribblesSuccessful/90', 'Dribble%', 'A2Recoveries/90', 'Aerial%', 'Tackle%',
                                       'TackInt/90'])
    DM = pd.DataFrame(data=df, columns=['Team', 'Player', 'Position', 'Nationality',
                                       'Passes/90','PassesComplete/90', 'KeyPasses/90',
                                       'Pass%', 'DribblesSuccessful/90', 'Dribble%', 'A2Recoveries/90', 'Aerial%', 'Tackle%',
                                       'TackInt/90'])

    D = pd.DataFrame(data=df, columns=['Team', 'Player', 'Position', 'Nationality', 'Passes/90', 'Pass%', 'Dribble%',
                                       'Recoveries/90', 'TackInt/90', 'Tackle%',
                                       'AerialsWon/90', 'Aerial%', 'D2LostBalls/90'])
    FW = FW[FW['Player'].isin(player)]
    test = FW[FW['Player'] == player]
    test = test.drop(columns=['Team', 'Position', 'Nationality'])
    test = test.set_index('Player')
    test = test.transpose()
    test.head()

    ax = test.plot.barh(figsize=(32, 24))
    ax.set_xlim([0, 1])
    ax.set_xticks([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    ax.xaxis.grid(True, linestyle='--', which='major',
                  color='black', alpha=.5)
    ax.axvline(.5, color='red', alpha=0.5, lw=6)  # median position
    ax.set_facecolor('#E6E6E6')
    plt.title(str(player) + '\nForwards Percentile Chart', weight='bold',size=24)
    st.pyplot(plt)

    playerdf = totaldf[totaldf['Player'].isin(player)]
    st.dataframe(playerdf)

def AMPercentile(state):
    df = load_data()

    df['Per90'] = df.Minutes / 90
    df['NPenGoals'] = df.Goals - df.Penalties
    df['TackInt'] = df['Successful Tackles'] + df['Interceptions']
    df['NPGoals/90'] = df.NPenGoals / df.Per90
    df['xG/90'] = df.xG / df.Per90
    df['xA/90'] = df.xA / df.Per90
    df['Assist/90'] = df.Assists / df.Per90
    df['Shots/90'] = df.Shots / df.Per90
    df['Passes/90'] = df.Passes / df.Per90
    df['PassesComplete/90'] = df['Passes Complete'] / df.Per90
    df['KeyPasses/90'] = df['Key Passes'] / df.Per90
    df['CrossesComplete/90'] = df['Crosses Complete'] / df.Per90
    df['D2LostBalls/90'] = df['D2 Lost Balls'] / df.Per90
    df['AerialsWon/90'] = df['Aerials Won'] / df.Per90
    df['DribblesSuccessful/90'] = df['Dribbles Successful'] / df.Per90
    df['TackInt/90'] = df['TackInt'] / df.Per90
    df['TacklesWon/90'] = df['Successful Tackles'] / df.Per90
    df['Interceptions/90'] = df.Interceptions / df.Per90
    df['A2Recoveries/90'] = df['A2 Recoveries'] / df.Per90
    df['Recoveries/90'] = df['Ball Recoveries'] / df.Per90

    totaldf = pd.DataFrame(data=df,
                      columns=['Team', 'Player', 'Position', 'Nationality', 'NPGoals/90', 'xG/90', 'Assist/90', 'xA/90',
                               'Shots/90',
                               'Shot%', 'Passes/90', 'PassesComplete/90', 'Pass%', 'KeyPasses/90', 'CrossesComplete/90',
                               'DribblesSuccessful/90', 'Dribble%', 'D2LostBalls/90', 'AerialsWon/90', 'Aerial%', 'Tackle%',
                               'TacklesWon/90',
                               'Interceptions/90', 'TackInt/90', 'A2Recoveries/90', 'Recoveries/90'])

    df = pd.DataFrame(data=df,
                      columns=['Team', 'Player', 'Position', 'Nationality', 'NPGoals/90', 'xG/90', 'Assist/90', 'xA/90',
                               'Shots/90',
                               'Shot%', 'Passes/90', 'PassesComplete/90', 'Pass%', 'KeyPasses/90', 'CrossesComplete/90',
                               'DribblesSuccessful/90', 'Dribble%', 'D2LostBalls/90', 'AerialsWon/90', 'Aerial%', 'Tackle%',
                               'TacklesWon/90',
                               'Interceptions/90', 'TackInt/90', 'A2Recoveries/90', 'Recoveries/90'])

    df['NPGoals/90'] = df['NPGoals/90'].rank(pct=True)
    df['xG/90'] = df['xG/90'].rank(pct=True)
    df['Assist/90'] = df['Assist/90'].rank(pct=True)
    df['xA/90'] = df['xA/90'].rank(pct=True)
    df['Shots/90'] = df['Shots/90'].rank(pct=True)
    df['Shot%'] = df['Shot%'].rank(pct=True)
    df['Passes/90'] = df['Passes/90'].rank(pct=True)
    df['PassesComplete/90'] = df['PassesComplete/90'].rank(pct=True)
    df['Pass%'] = df['Pass%'].rank(pct=True)
    df['KeyPasses/90'] = df['KeyPasses/90'].rank(pct=True)
    df['CrossesComplete/90'] = df['CrossesComplete/90'].rank(pct=True)
    df['DribblesSuccessful/90'] = df['DribblesSuccessful/90'].rank(pct=True)
    df['D2LostBalls/90'] = df['D2LostBalls/90'].rank(pct=True)
    df['AerialsWon/90'] = df['AerialsWon/90'].rank(pct=True)
    df['Aerial%'] = df['Aerial%'].rank(pct=True)
    df['Tackle%'] = df['Tackle%'].rank(pct=True)
    df['Dribble%'] = df['Dribble%'].rank(pct=True)
    df['TacklesWon/90'] = df['TacklesWon/90'].rank(pct=True)
    df['Interceptions/90'] = df['Interceptions/90'].rank(pct=True)
    df['TackInt/90'] = df['TackInt/90'].rank(pct=True)
    df['A2Recoveries/90'] = df['A2Recoveries/90'].rank(pct=True)
    df['Recoveries/90'] = df['Recoveries/90'].rank(pct=True)


    player = st.sidebar.multiselect('Select Player', natsorted(df.Player.unique()))

    FW = pd.DataFrame(data=df,
                      columns=['Team', 'Player', 'Position', 'Nationality', 'NPGoals/90', 'xG/90', 'Shots/90', 'Shot%',
                               'Assist/90', 'xA/90', 'KeyPasses/90',
                               'CrossesComplete/90', 'Pass%', 'DribblesSuccessful/90', 'A2Recoveries/90', 'Aerial%'])
    AM = pd.DataFrame(data=df, columns=['Team', 'Player', 'Position', 'Nationality', 'NPGoals/90', 'xG/90', 'Shots/90',
                                       'Assist/90', 'xA/90', 'Passes/90', 'KeyPasses/90',
                                       'Pass%', 'DribblesSuccessful/90', 'Dribble%', 'A2Recoveries/90', 'Aerial%',
                                       'TackInt/90'])
    DM = pd.DataFrame(data=df, columns=['Team', 'Player', 'Position', 'Nationality',
                                       'Passes/90','PassesComplete/90', 'KeyPasses/90',
                                       'Pass%', 'DribblesSuccessful/90', 'Dribble%', 'A2Recoveries/90', 'Aerial%', 'Tackle%',
                                       'TackInt/90'])

    D = pd.DataFrame(data=df, columns=['Team', 'Player', 'Position', 'Nationality', 'Passes/90', 'Pass%', 'Dribble%',
                                       'Recoveries/90', 'TackInt/90', 'Tackle%',
                                       'AerialsWon/90', 'Aerial%', 'D2LostBalls/90'])
    AM = AM[AM['Player'].isin(player)]
    test = AM[AM['Player'] == player]
    test = test.drop(columns=['Team', 'Position', 'Nationality'])
    test = test.set_index('Player')
    test = test.transpose()
    test.head()

    ax = test.plot.barh(figsize=(32, 24))
    ax.set_xlim([0, 1])
    ax.set_xticks([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    ax.xaxis.grid(True, linestyle='--', which='major',
                  color='black', alpha=.5)
    ax.axvline(.5, color='red', alpha=0.5, lw=6)  # median position
    ax.set_facecolor('#E6E6E6')
    plt.title(str(player) + '\nMidfield Percentile Chart', weight='bold',size=24)
    st.pyplot(plt)

    playerdf = totaldf[totaldf['Player'].isin(player)]
    st.dataframe(playerdf)

def DMPercentile(state):
    df = load_data()

    df['Per90'] = df.Minutes / 90
    df['NPenGoals'] = df.Goals - df.Penalties
    df['TackInt'] = df['Successful Tackles'] + df['Interceptions']
    df['NPGoals/90'] = df.NPenGoals / df.Per90
    df['xG/90'] = df.xG / df.Per90
    df['xA/90'] = df.xA / df.Per90
    df['Assist/90'] = df.Assists / df.Per90
    df['Shots/90'] = df.Shots / df.Per90
    df['Passes/90'] = df.Passes / df.Per90
    df['PassesComplete/90'] = df['Passes Complete'] / df.Per90
    df['KeyPasses/90'] = df['Key Passes'] / df.Per90
    df['CrossesComplete/90'] = df['Crosses Complete'] / df.Per90
    df['D2LostBalls/90'] = df['D2 Lost Balls'] / df.Per90
    df['AerialsWon/90'] = df['Aerials Won'] / df.Per90
    df['DribblesSuccessful/90'] = df['Dribbles Successful'] / df.Per90
    df['TackInt/90'] = df['TackInt'] / df.Per90
    df['TacklesWon/90'] = df['Successful Tackles'] / df.Per90
    df['Interceptions/90'] = df.Interceptions / df.Per90
    df['A2Recoveries/90'] = df['A2 Recoveries'] / df.Per90
    df['Recoveries/90'] = df['Ball Recoveries'] / df.Per90

    totaldf = pd.DataFrame(data=df,
                      columns=['Team', 'Player', 'Position', 'Nationality', 'NPGoals/90', 'xG/90', 'Assist/90', 'xA/90',
                               'Shots/90',
                               'Shot%', 'Passes/90', 'PassesComplete/90', 'Pass%', 'KeyPasses/90', 'CrossesComplete/90',
                               'DribblesSuccessful/90', 'Dribble%', 'D2LostBalls/90', 'AerialsWon/90', 'Aerial%', 'Tackle%',
                               'TacklesWon/90',
                               'Interceptions/90', 'TackInt/90', 'A2Recoveries/90', 'Recoveries/90'])

    df = pd.DataFrame(data=df,
                      columns=['Team', 'Player', 'Position', 'Nationality', 'NPGoals/90', 'xG/90', 'Assist/90', 'xA/90',
                               'Shots/90',
                               'Shot%', 'Passes/90', 'PassesComplete/90', 'Pass%', 'KeyPasses/90', 'CrossesComplete/90',
                               'DribblesSuccessful/90', 'Dribble%', 'D2LostBalls/90', 'AerialsWon/90', 'Aerial%', 'Tackle%',
                               'TacklesWon/90',
                               'Interceptions/90', 'TackInt/90', 'A2Recoveries/90', 'Recoveries/90'])

    df['NPGoals/90'] = df['NPGoals/90'].rank(pct=True)
    df['xG/90'] = df['xG/90'].rank(pct=True)
    df['Assist/90'] = df['Assist/90'].rank(pct=True)
    df['xA/90'] = df['xA/90'].rank(pct=True)
    df['Shots/90'] = df['Shots/90'].rank(pct=True)
    df['Shot%'] = df['Shot%'].rank(pct=True)
    df['Passes/90'] = df['Passes/90'].rank(pct=True)
    df['PassesComplete/90'] = df['PassesComplete/90'].rank(pct=True)
    df['Pass%'] = df['Pass%'].rank(pct=True)
    df['KeyPasses/90'] = df['KeyPasses/90'].rank(pct=True)
    df['CrossesComplete/90'] = df['CrossesComplete/90'].rank(pct=True)
    df['DribblesSuccessful/90'] = df['DribblesSuccessful/90'].rank(pct=True)
    df['D2LostBalls/90'] = df['D2LostBalls/90'].rank(pct=True)
    df['AerialsWon/90'] = df['AerialsWon/90'].rank(pct=True)
    df['Aerial%'] = df['Aerial%'].rank(pct=True)
    df['Tackle%'] = df['Tackle%'].rank(pct=True)
    df['Dribble%'] = df['Dribble%'].rank(pct=True)
    df['TacklesWon/90'] = df['TacklesWon/90'].rank(pct=True)
    df['Interceptions/90'] = df['Interceptions/90'].rank(pct=True)
    df['TackInt/90'] = df['TackInt/90'].rank(pct=True)
    df['A2Recoveries/90'] = df['A2Recoveries/90'].rank(pct=True)
    df['Recoveries/90'] = df['Recoveries/90'].rank(pct=True)


    player = st.sidebar.multiselect('Select Player', natsorted(df.Player.unique()))

    FW = pd.DataFrame(data=df,
                      columns=['Team', 'Player', 'Position', 'Nationality', 'NPGoals/90', 'xG/90', 'Shots/90', 'Shot%',
                               'Assist/90', 'xA/90', 'KeyPasses/90',
                               'CrossesComplete/90', 'Pass%', 'DribblesSuccessful/90', 'A2Recoveries/90', 'Aerial%'])
    AM = pd.DataFrame(data=df, columns=['Team', 'Player', 'Position', 'Nationality', 'NPGoals/90', 'xG/90', 'Shots/90',
                                       'Assist/90', 'xA/90', 'Passes/90', 'KeyPasses/90',
                                       'Pass%', 'DribblesSuccessful/90', 'Dribble%', 'A2Recoveries/90', 'Aerial%', 'Tackle%',
                                       'TackInt/90'])
    DM = pd.DataFrame(data=df, columns=['Team', 'Player', 'Position', 'Nationality',
                                       'Passes/90','PassesComplete/90', 'KeyPasses/90',
                                       'Pass%', 'DribblesSuccessful/90', 'Dribble%', 'A2Recoveries/90', 'Aerial%', 'Tackle%',
                                       'TackInt/90'])

    D = pd.DataFrame(data=df, columns=['Team', 'Player', 'Position', 'Nationality', 'Passes/90', 'Pass%', 'Dribble%',
                                       'Recoveries/90', 'TackInt/90', 'Tackle%',
                                       'AerialsWon/90', 'Aerial%', 'D2LostBalls/90'])
    DM = DM[DM['Player'].isin(player)]
    test = DM[DM['Player'] == player]
    test = test.drop(columns=['Team', 'Position', 'Nationality'])
    test = test.set_index('Player')
    test = test.transpose()
    test.head()

    ax = test.plot.barh(figsize=(32, 24))
    ax.set_xlim([0, 1])
    ax.set_xticks([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    ax.xaxis.grid(True, linestyle='--', which='major',
                  color='black', alpha=.5)
    ax.axvline(.5, color='red', alpha=0.5, lw=6)  # median position
    ax.set_facecolor('#E6E6E6')
    plt.title(str(player) + '\nDefensive Midfield Percentile Chart', weight='bold',size=24)
    st.pyplot(plt)

    playerdf = totaldf[totaldf['Player'].isin(player)]
    st.dataframe(playerdf)

def DPercentile(state):
    df = load_data()

    df['Per90'] = df.Minutes / 90
    df['NPenGoals'] = df.Goals - df.Penalties
    df['TackInt'] = df['Successful Tackles'] + df['Interceptions']
    df['NPGoals/90'] = df.NPenGoals / df.Per90
    df['xG/90'] = df.xG / df.Per90
    df['xA/90'] = df.xA / df.Per90
    df['Assist/90'] = df.Assists / df.Per90
    df['Shots/90'] = df.Shots / df.Per90
    df['Passes/90'] = df.Passes / df.Per90
    df['PassesComplete/90'] = df['Passes Complete'] / df.Per90
    df['KeyPasses/90'] = df['Key Passes'] / df.Per90
    df['CrossesComplete/90'] = df['Crosses Complete'] / df.Per90
    df['D2LostBalls/90'] = df['D2 Lost Balls'] / df.Per90
    df['AerialsWon/90'] = df['Aerials Won'] / df.Per90
    df['DribblesSuccessful/90'] = df['Dribbles Successful'] / df.Per90
    df['TackInt/90'] = df['TackInt'] / df.Per90
    df['TacklesWon/90'] = df['Successful Tackles'] / df.Per90
    df['Interceptions/90'] = df.Interceptions / df.Per90
    df['A2Recoveries/90'] = df['A2 Recoveries'] / df.Per90
    df['Recoveries/90'] = df['Ball Recoveries'] / df.Per90

    totaldf = pd.DataFrame(data=df,
                      columns=['Team', 'Player', 'Position', 'Nationality', 'NPGoals/90', 'xG/90', 'Assist/90', 'xA/90',
                               'Shots/90',
                               'Shot%', 'Passes/90', 'PassesComplete/90', 'Pass%', 'KeyPasses/90', 'CrossesComplete/90',
                               'DribblesSuccessful/90', 'Dribble%', 'D2LostBalls/90', 'AerialsWon/90', 'Aerial%', 'Tackle%',
                               'TacklesWon/90',
                               'Interceptions/90', 'TackInt/90', 'A2Recoveries/90', 'Recoveries/90'])

    df = pd.DataFrame(data=df,
                      columns=['Team', 'Player', 'Position', 'Nationality', 'NPGoals/90', 'xG/90', 'Assist/90', 'xA/90',
                               'Shots/90',
                               'Shot%', 'Passes/90', 'PassesComplete/90', 'Pass%', 'KeyPasses/90', 'CrossesComplete/90',
                               'DribblesSuccessful/90', 'Dribble%', 'D2LostBalls/90', 'AerialsWon/90', 'Aerial%', 'Tackle%',
                               'TacklesWon/90',
                               'Interceptions/90', 'TackInt/90', 'A2Recoveries/90', 'Recoveries/90'])

    df['NPGoals/90'] = df['NPGoals/90'].rank(pct=True)
    df['xG/90'] = df['xG/90'].rank(pct=True)
    df['Assist/90'] = df['Assist/90'].rank(pct=True)
    df['xA/90'] = df['xA/90'].rank(pct=True)
    df['Shots/90'] = df['Shots/90'].rank(pct=True)
    df['Shot%'] = df['Shot%'].rank(pct=True)
    df['Passes/90'] = df['Passes/90'].rank(pct=True)
    df['PassesComplete/90'] = df['PassesComplete/90'].rank(pct=True)
    df['Pass%'] = df['Pass%'].rank(pct=True)
    df['KeyPasses/90'] = df['KeyPasses/90'].rank(pct=True)
    df['CrossesComplete/90'] = df['CrossesComplete/90'].rank(pct=True)
    df['DribblesSuccessful/90'] = df['DribblesSuccessful/90'].rank(pct=True)
    df['D2LostBalls/90'] = df['D2LostBalls/90'].rank(pct=True)
    df['AerialsWon/90'] = df['AerialsWon/90'].rank(pct=True)
    df['Aerial%'] = df['Aerial%'].rank(pct=True)
    df['Tackle%'] = df['Tackle%'].rank(pct=True)
    df['Dribble%'] = df['Dribble%'].rank(pct=True)
    df['TacklesWon/90'] = df['TacklesWon/90'].rank(pct=True)
    df['Interceptions/90'] = df['Interceptions/90'].rank(pct=True)
    df['TackInt/90'] = df['TackInt/90'].rank(pct=True)
    df['A2Recoveries/90'] = df['A2Recoveries/90'].rank(pct=True)
    df['Recoveries/90'] = df['Recoveries/90'].rank(pct=True)


    player = st.sidebar.multiselect('Select Player', natsorted(df.Player.unique()))

    FW = pd.DataFrame(data=df,
                      columns=['Team', 'Player', 'Position', 'Nationality', 'NPGoals/90', 'xG/90', 'Shots/90', 'Shot%',
                               'Assist/90', 'xA/90', 'KeyPasses/90',
                               'CrossesComplete/90', 'Pass%', 'DribblesSuccessful/90', 'A2Recoveries/90', 'Aerial%'])
    AM = pd.DataFrame(data=df, columns=['Team', 'Player', 'Position', 'Nationality', 'NPGoals/90', 'xG/90', 'Shots/90',
                                       'Assist/90', 'xA/90', 'Passes/90', 'KeyPasses/90',
                                       'Pass%', 'DribblesSuccessful/90', 'Dribble%', 'A2Recoveries/90', 'Aerial%', 'Tackle%',
                                       'TackInt/90'])
    DM = pd.DataFrame(data=df, columns=['Team', 'Player', 'Position', 'Nationality',
                                       'Passes/90','PassesComplete/90', 'KeyPasses/90',
                                       'Pass%', 'DribblesSuccessful/90', 'Dribble%', 'A2Recoveries/90', 'Aerial%', 'Tackle%',
                                       'TackInt/90'])

    D = pd.DataFrame(data=df, columns=['Team', 'Player', 'Position', 'Nationality', 'Passes/90', 'Pass%', 'Dribble%',
                                       'Recoveries/90', 'TackInt/90', 'Tackle%',
                                       'AerialsWon/90', 'Aerial%', 'D2LostBalls/90'])
    D = D[D['Player'].isin(player)]
    test = D[D['Player'] == player]
    test = test.drop(columns=['Team', 'Position', 'Nationality'])
    test = test.set_index('Player')
    test = test.transpose()
    test.head()

    ax = test.plot.barh(figsize=(32, 24))
    ax.set_xlim([0, 1])
    ax.set_xticks([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    ax.xaxis.grid(True, linestyle='--', which='major',
                  color='black', alpha=.5)
    ax.axvline(.5, color='red', alpha=0.5, lw=6)  # median position
    ax.set_facecolor('#E6E6E6')
    plt.title(str(player) + '\nForwards Percentile Chart', weight='bold',size=24)
    st.pyplot(plt)

    playerdf = totaldf[totaldf['Player'].isin(player)]
    st.dataframe(playerdf)



if __name__ == "__main__":
    main()

#st.set_option('server.enableCORS', True)
# to run : streamlit run "/Users/michael/Documents/Python/Codes/NCAA Comparison App.py"


