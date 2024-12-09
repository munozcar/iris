# -*- coding: utf-8 -*-
# HITRAN MOLECULAR DATA PARSER


# --- THIS SECTION ADAPTED FROM CODE WRITTEN BY SIMON BRUDERER (2020) ----
"""
The class MolData reads and parses the molecular data

* Molecular data in the same format as for the Fortran 90 code are used. They consist of a block for the
  partition function and a block for the lines
* Both partition function and lines are stored as named tuples
* Only read access to the fields is granted through properties

- 01/06/2020: SB, initial version

"""

from collections import namedtuple, OrderedDict
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None
    pass

__all__ = ["MolData"]


class MolData:
    _partition_type = namedtuple('partition', ['t', 'q'])
    _lines_type = namedtuple('lines', ['nr', 'lev_up', 'lev_low', 'lam', 'freq', 'a_stein',
                                       'e_up', 'e_low', 'g_up', 'g_low'])

    def __init__(self, name, fname):
        """Initialize a molecular data structure and read from a .par file. The file has the same structure as in the
        Fortran 90 code, see notes below.

        Parameters
        ----------
        name: str
            Label of the molecule (e.g. "CO")
        fname: str
            Path/filename of the .par file

        Notes
        -----

        The file format of .par file is as follows:

        # Some comments, followed by the number of partition function values
        2933
        # Some further comments, followed by the partition function (temperature vs Q(temperature))
        70.0    25.544319
        71.0    25.904587
        ...
        # Some further comments, followed by the number of lines
        754
        # Some further comments followed by the lines
        1      1_R_13         1_R_13  196.29472  1527.25685208   1.3500e-04     3565.78714     3492.49079   58.0   54.0
        2      0_R_13         0_R_13  194.54560  1540.98815610   2.3660e-04      554.97254      481.01720   58.0   54.0

        Important: The lines section on the bottom is *fixed format*:
        - Nr: Index of the line - integer, 6 characters
        - Lev_up: Upper level label - string, 15 characters
        - Lev_low: Lower level label - string, 15 characters
        - Lam: Wavelength in micron - float, 11 characters
        - Freq: Frequency in GHz - float, 15 characters
        - A_stein: Einstein-A in s**-1 - float, 13 characters
        - E_up: Upper level energy in K - float, 15 characters
        - E_low: Lower level energy in K - float, 15 characters
        - g_up: Upper level statistical weight - float, 7 characters
        - g_low: Lower level statistical weight - float, 7 characters
        """

        self._name = name
        self._fname = fname

        self._partition, self._lines = self._read_molecule(self._fname)

    def _read_molecule(self, fname):
        """Reads the molecular data file, should be only called at initialization

        Parameters
        ----------
        fname: str
            Path/filename of the .par file

        Returns
        -------
        _partition_type:
            Partition function
        _lines_type:
            Lines
        """

        # 1. read file and split into block for partition function and lines
        with open(fname, "r") as f:
            data_raw = f.readlines()

        # remove comments and empty lines
        data_clean = list(filter(lambda x: len(x.strip()) > 0 and x.strip()[0] != "#", data_raw))

        # split file into partition function and line section
        n_partition = int(data_clean[0])

        data_q = data_clean[1:n_partition + 1]
        data_lines = data_clean[n_partition + 2:]

        # 2. read partition function
        N = np.genfromtxt(data_q, dtype="f8,f8")
        q_temperature, q = [N[field] for field in N.dtype.names]

        # 3. read lines

        # noinspection PyTypeChecker
        M = np.genfromtxt(data_lines, dtype="i4,S30,S30,f8,f8,f8,f8,f8,f8,f8",
                          delimiter=(6, 30, 30, 11, 15, 13, 15, 15, 7, 7))

        nr, lev_up, lev_low, lam, freq, a_stein, e_up, e_low, g_up, g_low = [M[field] for field in M.dtype.names]

        lev_up = list(map(self._decode_strip, lev_up))
        lev_low = list(map(self._decode_strip, lev_low))

        # convert frequency from GHz to Hz
        freq = 1e9 * freq

        # 4. return named tuples
        return MolData._partition_type(q_temperature, q), \
            MolData._lines_type(nr, lev_up, lev_low, lam, freq, a_stein, e_up, e_low, g_up, g_low)

    def __repr__(self):
        return f"MolData(name={self.name}, fname={self.fname}, {self.partition}, {self.lines})"

    @staticmethod
    def _decode_strip(x):
        return x.decode().strip()

    @property
    def name(self):
        """str: Name of the molecule"""
        return self._name

    @property
    def fname(self):
        """str: Path/filename of the molecule read in"""
        return self._fname

    @property
    def partition(self):
        """_partition_type: Partition function of the molecule"""
        return self._partition

    @property
    def lines(self):
        """_lines_type: Lines of the molecule"""
        return self._lines

    @property
    def get_table_lines(self):
        """pd.Dataframe: Pandas dataframe with lines table"""

        if pd is None:
            raise ImportError("Pandas required to create table")

        return pd.DataFrame({'nr': self.lines.nr,
                             'lev_up': self.lines.lev_up,
                             'lev_low': self.lines.lev_low,
                             'lam': self.lines.lam,
                             'freq': self.lines.freq,
                             'a_stein': self.lines.a_stein,
                             'e_up': self.lines.e_up,
                             'e_low': self.lines.e_low,
                             'g_up': self.lines.g_up,
                             'g_low': self.lines.g_low})

    @property
    def get_table_partition(self):
        """pd.Dataframe: Pandas dataframe with partition function table"""

        if pd is None:
            raise ImportError("Pandas required to create table")

        return pd.DataFrame({'t': self.partition.t,
                             'q': self.partition.q})

    def _repr_html_(self):
        # noinspection PyProtectedMember
        return self.get_table_lines._repr_html_()
# --- end ----

# Set up with iris (CEM-R)
def setup_catalog(molecule_names, wlow, whigh, wpad=0.01, path='./'):
    path_to_moldata = path+'/'
    # initialize dictionary
    parameters = OrderedDict()
    levels = OrderedDict()
    
    for mol_name in molecule_names:
        
        keys = [x.partition('_')[0] for x in parameters.keys()]
        component_count = keys.count(mol_name)
        # initialize dictionary
        mol_name = mol_name + '_{}'.format(component_count)
        parameters[mol_name] = {}
        levels[mol_name] = {}
        
        # read file   
        mol_data = MolData(mol_name.partition('_')[0], path_to_moldata+"data_Hitran_2020_{}.par".format(mol_name.partition('_')[0]))
        # wavelength range
        parameters[mol_name]['wlow'] = wlow-wpad
        parameters[mol_name]['whigh'] = whigh+wpad
        # strucuture the line information
        parameters[mol_name]['ws'] = np.array([w for w in mol_data.get_table_lines['lam']])[::-1]
        levels[mol_name]['lev_up'] = np.array([w for w in mol_data.get_table_lines['lev_up']])[::-1]
        levels[mol_name]['lev_low'] = np.array([w for w in mol_data.get_table_lines['lev_low']])[::-1]
        parameters[mol_name]['aijs'] =  np.array([w for w in mol_data.get_table_lines['a_stein']])[::-1]
        parameters[mol_name]['eups'] = np.array([w for w in mol_data.get_table_lines['e_up']])[::-1] 
        parameters[mol_name]['elow'] = np.array([w for w in mol_data.get_table_lines['e_low']])[::-1] 
        parameters[mol_name]['gups'] = np.array([w for w in mol_data.get_table_lines['g_up']])[::-1] 
        parameters[mol_name]['glow'] = np.array([w for w in mol_data.get_table_lines['g_low']])[::-1] 
        parameters[mol_name]['nus'] = np.array([w for w in mol_data.get_table_lines['freq']])[::-1] 
        # apply wavelength range
        parameters[mol_name]['aijs'] = parameters[mol_name]['aijs'][(parameters[mol_name]['ws']>parameters[mol_name]['wlow'])*(parameters[mol_name]['ws']<parameters[mol_name]['whigh'])]
        levels[mol_name]['lev_up'] = levels[mol_name]['lev_up'][(parameters[mol_name]['ws']>parameters[mol_name]['wlow'])*(parameters[mol_name]['ws']<parameters[mol_name]['whigh'])]
        levels[mol_name]['lev_low'] = levels[mol_name]['lev_low'][(parameters[mol_name]['ws']>parameters[mol_name]['wlow'])*(parameters[mol_name]['ws']<parameters[mol_name]['whigh'])]
        parameters[mol_name]['eups'] = parameters[mol_name]['eups'][(parameters[mol_name]['ws']>parameters[mol_name]['wlow'])*(parameters[mol_name]['ws']<parameters[mol_name]['whigh'])]
        parameters[mol_name]['gups'] = parameters[mol_name]['gups'][(parameters[mol_name]['ws']>parameters[mol_name]['wlow'])*(parameters[mol_name]['ws']<parameters[mol_name]['whigh'])]
        parameters[mol_name]['elow'] = parameters[mol_name]['elow'][(parameters[mol_name]['ws']>parameters[mol_name]['wlow'])*(parameters[mol_name]['ws']<parameters[mol_name]['whigh'])]
        parameters[mol_name]['glow'] = parameters[mol_name]['glow'][(parameters[mol_name]['ws']>parameters[mol_name]['wlow'])*(parameters[mol_name]['ws']<parameters[mol_name]['whigh'])]
        parameters[mol_name]['nus'] = parameters[mol_name]['nus'][(parameters[mol_name]['ws']>parameters[mol_name]['wlow'])*(parameters[mol_name]['ws']<parameters[mol_name]['whigh'])]
        parameters[mol_name]['ws'] = parameters[mol_name]['ws'][(parameters[mol_name]['ws']>parameters[mol_name]['wlow'])*(parameters[mol_name]['ws']<parameters[mol_name]['whigh'])]
        # partition function
        parameters[mol_name]['Qt'] = mol_data.partition[0]
        parameters[mol_name]['Qv'] = mol_data.partition[1]    
        # de-allocate memory
        mol_data = None
    # de-allocate memory    
    mol_data = None
    
    return parameters, levels