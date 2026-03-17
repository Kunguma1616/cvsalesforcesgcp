from fastapi import APIRouter, HTTPException, Query
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from salesforce_servcice import SalesforcesService
from typing import Optional

routes = APIRouter(prefix="/api/ranking", tags=["ranking"])

# Keywords that identify a WET trade
WET_KEYWORDS = [
    "drain", "plumb", "gas", "water", "leak", "pipe",
    "boiler", "heating", "sanit", "sewer", "wet"
]


def _derive_trade_type(primary_trade: str, wet_flag, dry_flag) -> str:
    """
    Determine WET / DRY from SF boolean flags first.
    If both are null/False, fall back to keyword matching on the trade name.
    """
    if wet_flag:
        return "WET"
    if dry_flag:
        return "DRY"
    # Fallback: keyword match on trade name
    trade_lower = (primary_trade or "").lower()
    for kw in WET_KEYWORDS:
        if kw in trade_lower:
            return "WET"
    if trade_lower:
        return "DRY"
    return "N/A"


def get_engineer_rankings(sf, trade_group: Optional[str] = None):
    """Fetch engineer applications and return a ranked list."""
    print("[get_engineer_rankings] Fetching engineer rankings...")

    where_clause = ""
    if trade_group and trade_group.lower() != "all":
        safe_trade = trade_group.replace("'", "\\'")
        where_clause = f"WHERE Primary_Trade__c = '{safe_trade}'"

    query = f"""
        SELECT Id, First_Name__c, Last_Name__c, Email_Address__c,
               Primary_Trade__c, Wet_Trade__c, Dry_Trade__c
        FROM Engineer_Application__c
        {where_clause}
        ORDER BY CreatedDate DESC
    """
    print(f"[get_engineer_rankings] Executing: {query.strip()}")
    result = sf.excute_soql(query)

    engineers = []
    if result:
        for idx, row in enumerate(result, start=1):
            first = row.get("First_Name__c", "") or ""
            last  = row.get("Last_Name__c",  "") or ""
            name  = f"{first} {last}".strip() or "Unknown"
            trade = row.get("Primary_Trade__c", "") or "Unknown"

            engineers.append({
                "rank":        idx,
                "id":          row.get("Id", ""),
                "name":        name,
                "email":       row.get("Email_Address__c", "") or "",
                "trade_group": trade,
                "trade_type":  _derive_trade_type(
                    trade,
                    row.get("Wet_Trade__c", False),
                    row.get("Dry_Trade__c", False),
                ),
            })
    return engineers


def get_distinct_trade_groups(sf):
    """Get distinct trade groups from Engineer_Application__c."""
    print("[get_distinct_trade_groups] Fetching trade groups...")
    try:
        query = """
            SELECT Primary_Trade__c
            FROM Engineer_Application__c
            WHERE Primary_Trade__c != NULL
            GROUP BY Primary_Trade__c
            ORDER BY Primary_Trade__c ASC
        """
        result = sf.excute_soql(query)
        return [row["Primary_Trade__c"] for row in (result or []) if row.get("Primary_Trade__c")]
    except Exception as e:
        print(f"[get_distinct_trade_groups] Error: {e}")
        return []


# ── Routes ────────────────────────────────────────────────────────────────────

@routes.get("/engineers")
def get_ranked_engineers(
    trade_group: Optional[str] = Query(None, description="Filter by trade group")
):
    """Returns ranked list of engineer applications, optionally filtered by trade_group."""
    try:
        sf = SalesforcesService()
        engineers = get_engineer_rankings(sf, trade_group)
        return {"total": len(engineers), "engineers": engineers}
    except Exception as e:
        print(f"[/ranking/engineers] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@routes.get("/trade-groups")
def get_trade_groups():
    """Returns distinct trade groups for filter dropdown."""
    try:
        sf = SalesforcesService()
        trades = get_distinct_trade_groups(sf)
        return {"trade_groups": trades}
    except Exception as e:
        print(f"[/ranking/trade-groups] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
