import numpy as np
import pandas as pd
from typing import Literal

HIGH_DEMAND = 1e12  # Um valor alto para representar uma demanda alta


class TransportationOptimizer:
    """
    Uma classe para otimizar o problema do transporte.

    Esta classe usa uma abordagem baseada em custos para alocar as capacidades de produção de forma a atender às demandas com custos mínimos. Ela funciona com base no princípio do Método de Aproximação de Vogel (VAM).

    Atributos:
        production (np.ndarray): Um array que representa a capacidade de produção de cada fornecedor.
        demand (np.ndarray): Um array que representa a demanda necessária para cada consumidor.
        cost_matrix (np.ndarray): Uma matriz que representa o custo de transporte de cada fornecedor para cada consumidor.
        total_cost (float): O custo total após o processo de otimização.
        allocation_matrix (np.ndarray): Uma matriz que representa a alocação de produção para atender à demanda.
    """

    def __init__(
        self, production: np.ndarray, demand: np.ndarray, cost_matrix: np.ndarray
    ):
        """
        Inicializa o TransportationOptimizer com capacidades de produção, demanda e matriz de custos.

        Args:
            production (np.ndarray): Capacidades de produção para cada fornecedor.
            demand (np.ndarray): Demanda requerida para cada consumidor.
            cost_matrix (np.ndarray): Custo de transporte de cada fornecedor para cada consumidor.
        """

        self.production = production
        self.demand = demand
        self.cost_matrix = cost_matrix
        self.total_cost = 0
        self.allocation_matrix = np.zeros(cost_matrix.shape)

    def optimize(self) -> tuple[float, pd.DataFrame]:
        """
        Otimiza o plano de transporte para minimizar o custo total.

        Ele aloca iterativamente o suprimento para a demanda com base na abordagem de menor custo até que
        toda a demanda e suprimento sejam alocados.

        Returns:
            Uma tupla contendo o custo total e um DataFrame Pandas representando a matriz de alocação.
        """
        if np.sum(self.production) != np.sum(self.demand):
            raise ValueError("Production and demand must be equal.")

        while np.sum(self.production) > 0 and np.sum(self.demand) > 0:
            self._process_cost_allocation()

        return self.total_cost, pd.DataFrame(self.allocation_matrix)

    def _process_cost_allocation(self):
        """
        Processa uma única alocação de suprimento para demanda com base na diferença de custo.

        Ele determina se a alocação deve ser feita por linha ou coluna com base na maior diferença de custo.
        """
        row_diff, col_diff = self._calculate_cost_difference(self.cost_matrix)
        max_row_diff_index = np.argmax(row_diff)
        max_col_diff_index = np.argmax(col_diff)

        if row_diff[max_row_diff_index] >= col_diff[max_col_diff_index]:
            self._allocate("row", max_row_diff_index)
        else:
            self._allocate("column", max_col_diff_index)

    def _calculate_cost_difference(
        self, matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calcula a diferença de custo para cada linha e coluna na matriz de custos.

        Args:
            matrix (np.ndarray): A matriz de custos.

        Returns:
            Uma tupla de arrays contendo as diferenças de custo para as linhas e colunas.
        """
        row_diff = np.array([np.diff(np.partition(row, 1)[:2]) for row in matrix])
        col_diff = np.array([np.diff(np.partition(col, 1)[:2]) for col in matrix.T])
        return row_diff, col_diff

    def _allocate(self, axis: Literal["row", "column"], index: int):
        """
        Aloca o suprimento para a demanda, seja por linha ou coluna.

        Args:
            axis (Literal["row", "column"]): O eixo a ser alocado.
            index (int): O índice da linha ou coluna a ser alocada.
        """

        if axis == "row":
            self._allocate_row(index)
        else:
            self._allocate_column(index)

    def _allocate_row(self, row_index: int):
        """
        Aloca o suprimento para a demanda de uma linha específica.

        Args:
            row_index (int): O índice da linha a ser alocada.
        """
        min_cost_index = np.argmin(self.cost_matrix[row_index])
        allocated_amount = min(self.production[row_index], self.demand[min_cost_index])
        self._update_costs_and_allocation(row_index, min_cost_index, allocated_amount)

    def _allocate_column(self, col_index: int):
        """
        Aloca o suprimento para a demanda de uma coluna específica.

        Args:
            col_index (int): O índice da coluna a ser alocada.
        """
        min_cost_index = np.argmin(self.cost_matrix[:, col_index])
        allocated_amount = min(self.production[min_cost_index], self.demand[col_index])
        self._update_costs_and_allocation(min_cost_index, col_index, allocated_amount)

    def _update_costs_and_allocation(
        self, row_index: int, col_index: int, amount: float
    ):
        """
        Atualiza o custo total, produção e demanda após uma alocação.

        Args:
            row_index (int): O índice da linha onde a alocação é feita.
            col_index (int): O índice da coluna onde a alocação é feita.
            amount (float): A quantidade alocada.
        """
        self.total_cost += self.cost_matrix[row_index, col_index] * amount
        self.production[row_index] -= amount
        self.demand[col_index] -= amount
        self.allocation_matrix[row_index, col_index] += amount

        self._update_cost_matrix(row_index, col_index)

    def _update_cost_matrix(self, row_index: int, col_index: int):
        """
        Atualiza a matriz de custos após uma alocação.
        Args:
            row_index (int): O índice da linha onde a alocação é feita.
            col_index (int): O índice da coluna onde a alocação é feita.
        """
        if self.production[row_index] == 0:
            self.cost_matrix[row_index, :] = HIGH_DEMAND
        if self.demand[col_index] == 0:
            self.cost_matrix[:, col_index] = HIGH_DEMAND


routes = ["Linha Central", "Rota do Parque", "Via Expressa"]
capacities = np.array(
    [1250, 1550, 1400]
)  # Número máximo de passageiros por dia, para cada rota

neighborhoods = ["Centro", "Parque das Flores", "Vila Nova", "Jardim Oceânico"]
demands = np.array(
    [1200, 950, 750, 1300]
)  # Número de passageiros por dia, para cada bairro
# Custo operacional por passageiro em cada rota e bairro (estimativa)
# Unidade: Custo em reais por passageiro
costs = np.array(
    [
        [3.50, 4.00, 3.75, 4.50],  # Custos para "Linha Central"
        [3.00, 3.50, 4.00, 3.60],  # Custos para "Rota do Parque"
        [3.20, 4.20, 3.90, 4.10],  # Custos para "Via Expressa"
    ]
)

TransportationOptimizer(capacities, demands, costs).optimize()