from fastapi import APIRouter, HTTPException
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from salesforce_servcice import SalesforcesService

routes = APIRouter(prefix="/api/dashboad", tags=["dashboad"])


# ── Total counts ──────────────────────────────────────────────────────────────

def get_enginners_appiction_count(sf):
    """Total applications where Wet OR Dry trade is TRUE."""
    print("[get_enginners_appiction_count] Fetching total application count...")
    try:
        query = """
            SELECT COUNT(Id)
            FROM Engineer_Application__c
            WHERE Wet_Trade__c = TRUE
               OR Dry_Trade__c = TRUE
        """
        print(f"[get_enginners_appiction_count] Executing: {query.strip()}")
        result = sf.excute_soql(query)
        print(f"[get_enginners_appiction_count] Result: {result}")
        if result:
            return result[0].get("expr0", 0)
        return 0
    except Exception as e:
        print(f"[get_enginners_appiction_count] Error: {e}")
        return 0


def get_total_WET_Trade_applications(sf):
    """Total count of WET-Trade applications."""
    print("[get_total_WET_Trade_applications] Fetching total WET-Trade count...")
    try:
        query = """
            SELECT COUNT(Id)
            FROM Engineer_Application__c
            WHERE Wet_Trade__c = TRUE
        """
        print(f"[get_total_WET_Trade_applications] Executing: {query.strip()}")
        result = sf.excute_soql(query)
        print(f"[get_total_WET_Trade_applications] Result: {result}")
        if result and len(result) > 0:
            return result[0].get("expr0", 0)
        return 0
    except Exception as e:
        print(f"[get_total_WET_Trade_applications] Error: {e}")
        return 0


def get_total_DRY_Trade_applications(sf):
    """Total count of DRY-Trade applications (excludes Wet)."""
    print("[get_total_DRY_Trade_applications] Fetching total DRY-Trade count...")
    try:
        query = """
            SELECT COUNT(Id)
            FROM Engineer_Application__c
            WHERE Dry_Trade__c = TRUE
              AND Wet_Trade__c = FALSE
        """
        print(f"[get_total_DRY_Trade_applications] Executing: {query.strip()}")
        result = sf.excute_soql(query)
        print(f"[get_total_DRY_Trade_applications] Result: {result}")
        if result and len(result) > 0:
            return result[0].get("expr0", 0)
        return 0
    except Exception as e:
        print(f"[get_total_DRY_Trade_applications] Error: {e}")
        return 0


# ── Breakdown by Primary_Trade__c ─────────────────────────────────────────────

def get_WET_Trade_breakdown(sf):
    """
    Breakdown of WET-Trade applications grouped by Primary_Trade__c.

    Query:
        SELECT Primary_Trade__c, COUNT(Id)
        FROM Engineer_Application__c
        WHERE Wet_Trade__c = TRUE
        GROUP BY Primary_Trade__c
        ORDER BY COUNT(Id) DESC
    """
    print("[get_WET_Trade_breakdown] Fetching WET-Trade breakdown by Primary Trade...")
    try:
        query = """
            SELECT Primary_Trade__c, COUNT(Id)
            FROM Engineer_Application__c
            WHERE Wet_Trade__c = TRUE
            GROUP BY Primary_Trade__c
            ORDER BY COUNT(Id) DESC
        """
        print(f"[get_WET_Trade_breakdown] Executing: {query.strip()}")
        result = sf.excute_soql(query)
        print(f"[get_WET_Trade_breakdown] Result: {result}")

        breakdown = []
        if result:
            for row in result:
                breakdown.append({
                    "primary_trade": row.get("Primary_Trade__c", "Unknown"),
                    "count": row.get("expr0", 0)
                })
        return breakdown
    except Exception as e:
        print(f"[get_WET_Trade_breakdown] Error: {e}")
        return []


def get_DRY_Trade_breakdown(sf):
    """
    Breakdown of DRY-Trade applications grouped by Primary_Trade__c.
    Excludes records that are also marked as Wet.

    Query:
        SELECT Primary_Trade__c, COUNT(Id)
        FROM Engineer_Application__c
        WHERE Dry_Trade__c = TRUE
          AND Wet_Trade__c = FALSE
        GROUP BY Primary_Trade__c
        ORDER BY COUNT(Id) DESC
    """
    print("[get_DRY_Trade_breakdown] Fetching DRY-Trade breakdown by Primary Trade...")
    try:
        query = """
            SELECT Primary_Trade__c, COUNT(Id)
            FROM Engineer_Application__c
            WHERE Dry_Trade__c = TRUE
              AND Wet_Trade__c = FALSE
            GROUP BY Primary_Trade__c
            ORDER BY COUNT(Id) DESC
        """
        print(f"[get_DRY_Trade_breakdown] Executing: {query.strip()}")
        result = sf.excute_soql(query)
        print(f"[get_DRY_Trade_breakdown] Result: {result}")

        breakdown = []
        if result:
            for row in result:
                breakdown.append({
                    "primary_trade": row.get("Primary_Trade__c", "Unknown"),
                    "count": row.get("expr0", 0)
                })
        return breakdown
    except Exception as e:
        print(f"[get_DRY_Trade_breakdown] Error: {e}")
        return []


# ── Routes ────────────────────────────────────────────────────────────────────

@routes.get("/stats")
def get_dashboard_stats():
    """
    Returns all dashboard stats:
    - total applications
    - total wet / dry counts
    - wet & dry breakdown by Primary_Trade__c
    """
    try:
        from salesforce_servcice import SalesforcesService
        sf = SalesforcesService()

        total          = get_enginners_appiction_count(sf)
        total_wet      = get_total_WET_Trade_applications(sf)
        total_dry      = get_total_DRY_Trade_applications(sf)
        wet_breakdown  = get_WET_Trade_breakdown(sf)
        dry_breakdown  = get_DRY_Trade_breakdown(sf)

        return {
            "total_applications":  total,
            "total_wet_trade":     total_wet,
            "total_dry_trade":     total_dry,
            "wet_trade_breakdown": wet_breakdown,
            "dry_trade_breakdown": dry_breakdown,
        }

    except Exception as e:
        print(f"[/stats] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))