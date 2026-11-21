import logging
import time
import unicodedata
from abc import ABC, abstractmethod

from app.models.travel import AgentResult, AgentStatus, TravelPlan
from app.providers.http import ProviderError

logger = logging.getLogger(__name__)


def normalise_name(name: str) -> str:
    """Matching key for place names: "Belém Tower" and "belem tower" compare equal."""
    decomposed = unicodedata.normalize("NFKD", name.casefold())
    return "".join(ch for ch in decomposed if ch.isalnum())


class AgentSkipped(Exception):
    """Raised when an agent cannot run because a prerequisite is unavailable.

    Distinct from a failure: the pipeline continues and the reason is surfaced
    to the user as a warning rather than an error.
    """


class BaseAgent(ABC):
    name: str = "BaseAgent"
    description: str = ""
    source: str = ""

    @abstractmethod
    async def run(self, plan: TravelPlan) -> TravelPlan:
        """Enrich the plan in place and return it."""

    async def execute(self, plan: TravelPlan) -> tuple[TravelPlan, AgentResult]:
        """Run the agent, capturing timing and converting errors into results."""
        start = time.perf_counter()

        def elapsed_ms() -> int:
            return int((time.perf_counter() - start) * 1000)

        try:
            plan = await self.run(plan)
            result = AgentResult(
                agent_name=self.name,
                status=AgentStatus.COMPLETED,
                message=self.success_message(plan),
                duration_ms=elapsed_ms(),
                source=self.source,
            )
        except AgentSkipped as exc:
            result = AgentResult(
                agent_name=self.name,
                status=AgentStatus.SKIPPED,
                message=str(exc),
                duration_ms=elapsed_ms(),
                source=self.source,
            )
            plan.warnings.append(f"{self.name}: {exc}")
        except ProviderError as exc:
            logger.warning("%s provider error: %s", self.name, exc)
            result = AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                message=f"Data source unavailable: {exc}",
                duration_ms=elapsed_ms(),
                source=self.source,
            )
            plan.warnings.append(f"{self.name}: {exc}")
        except Exception as exc:
            logger.exception("%s crashed", self.name)
            result = AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                message=f"{type(exc).__name__}: {exc}",
                duration_ms=elapsed_ms(),
                source=self.source,
            )
            plan.warnings.append(f"{self.name} failed: {exc}")

        plan.agent_results.append(result)
        return plan, result

    def success_message(self, plan: TravelPlan) -> str:
        return f"{self.name} completed."
