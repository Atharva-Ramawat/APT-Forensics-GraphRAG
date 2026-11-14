"""
Phase 3: FalkorDB Graph Client for Causal Topology & Blast Radius Traversal
Connects to AWS FalkorDB instance using credentials from .env
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load credentials from .env (reusing existing NEO_FRONTIER configuration)
load_dotenv()

FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", 6379))
FALKORDB_PASSWORD = os.getenv("FALKORDB_PASSWORD", None)
FALKORDB_USERNAME = os.getenv("FALKORDB_USERNAME", None)
FALKORDB_GRAPH_NAME = os.getenv("FALKORDB_GRAPH_NAME", "apt_threat_graph")


class FalkorDBClient:
    """Manages connection and Cypher graph queries to AWS FalkorDB."""

    def __init__(self):
        self.driver = None
        self.graph = None
        self._initialize_connection()

    def _initialize_connection(self):
        """Initializes connection to FalkorDB."""
        try:
            from falkordb import FalkorDB
            self.driver = FalkorDB(
                host=FALKORDB_HOST,
                port=FALKORDB_PORT,
                password=FALKORDB_PASSWORD,
                username=FALKORDB_USERNAME
            )
            self.graph = self.driver.select_graph(FALKORDB_GRAPH_NAME)
            print(f"[+] Connected to FalkorDB on {FALKORDB_HOST}:{FALKORDB_PORT} [Graph: {FALKORDB_GRAPH_NAME}]")
        except Exception as e:
            print(f"[!] Warning: FalkorDB connection failed ({e}). Operating in disconnected/fallback mode.")
            self.driver = None
            self.graph = None

    def query_blast_radius(self, ip_address: str, hops: int = 3) -> str:
        """
        Executes an openCypher query traversing 1 to `hops` outward from a compromised IP.
        Returns a formatted plain-text string representation of the discovered topology.
        """
        if not self.graph:
            # Fallback mock topology if database is currently unreachable
            return (
                f"[FalkorDB Offline / Mock Response]\n"
                f"Blast Radius for IP: {ip_address} (Up to {hops} hops):\n"
                f"└── (Entity: IP_{ip_address})\n"
                f"    └── -[failed_authentication_to]-> (Host: NGINX_Gateway_10.0.0.5)\n"
                f"        └── -[lateral_connection_to]-> (Service: Auth_Service_10.0.1.14)\n"
                f"            └── -[granted_iam_role]-> (Role: RDS_ReadOnly_Access)"
            )

        cypher_query = f"""
        MATCH path = (start)-[r*1..{hops}]->(target)
        WHERE start.id = $ip OR start.ip = $ip OR start.name = $ip
        RETURN path
        LIMIT 50
        """

        try:
            result = self.graph.query(cypher_query, {"ip": ip_address})
            
            if not result.result_set:
                return f"No outward relationships or downstream dependencies found for IP: {ip_address} within {hops} hops."

            topology_lines = [f"=== Discovered Blast Radius Topology for {ip_address} ({hops} Hops) ==="]
            for row in result.result_set:
                path = row[0]
                topology_lines.append(f"• {str(path)}")

            return "\n".join(topology_lines)

        except Exception as query_err:
            return f"[!] Error executing Cypher blast radius query for {ip_address}: {str(query_err)}"


# Singleton instance for application-wide imports
falkor_client = FalkorDBClient()


def query_blast_radius(ip_address: str, hops: int = 3) -> str:
    """Convenience helper function for the FastAPI endpoint."""
    return falkor_client.query_blast_radius(ip_address=ip_address, hops=hops)


if __name__ == "__main__":
    test_ip = "10.0.0.5"
    print(f"[*] Testing blast radius query for {test_ip}...")
    print(query_blast_radius(test_ip, hops=3))
