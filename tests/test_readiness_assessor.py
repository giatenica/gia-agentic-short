"""Readiness Assessor Unit Tests.

Focused tests to ensure readiness assessment integration does not crash when
workflow results are partially populated (some agent results can be None).

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import pytest
from unittest.mock import MagicMock, patch

from src.agents.readiness_assessor import ReadinessAssessorAgent
from src.utils.time_tracking import TimeTrackingReport


class TestReadinessAssessorTimeTracking:
    @pytest.mark.unit
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
    @patch('src.llm.claude_client.anthropic.Anthropic')
    @patch('src.llm.claude_client.anthropic.AsyncAnthropic')
    def test_update_time_tracking_skips_none_agent_entries(self, mock_async_anthropic, mock_anthropic):
        agent = ReadinessAssessorAgent()

        report = TimeTrackingReport(project_id="test", project_folder="/tmp/test")

        workflow_results = {
            "total_time": 12.3,
            "total_tokens": 45,
            "agents": {
                "data_analysis": {
                    "agent_name": "DataAnalyst",
                    "execution_time": 1.25,
                    "tokens_used": 10,
                },
                "consistency_check": None,
                "readiness_assessment": None,
            },
        }

        # Should not raise
        agent._update_time_tracking(report, workflow_results)

        assert len(report.workflow_executions) == 1
        execution = report.workflow_executions[0]
        assert execution["total_time"] == 12.3
        assert execution["total_tokens"] == 45
        assert execution["agent_times"].get("DataAnalyst") == 1.25
