{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TupleSections #-}


-- | Top-level sentence representation, as well as its transformation to a
-- graph (tree) representation.


module MWE.Sent
  ( Sent (..)
  , cleanUp
  , dummyRootPOS
  , dummyRootDepRel

  , Node (..)
  , toGraph
  , tokID
  ) where


import           GHC.TypeNats (KnownNat)
import           GHC.Generics (Generic)

import           Control.Monad (guard)
import           Numeric.LinearAlgebra.Static.Backprop (R)

import           Data.Binary (Binary)
import qualified Data.List as List
-- import qualified Data.Set as S
import qualified Data.Map.Strict as M
import qualified Data.Graph as G
import qualified Data.Text as T

import qualified Format.Cupt as Cupt
import qualified Graph as Graph


----------------------------------------------
-- Sentence
----------------------------------------------


-- | Input sentence
data Sent d = Sent
  { cuptSent :: Cupt.Sent
    -- ^ The .cupt sentence
  , wordEmbs :: [R d]
    -- ^ The corresponding word embeddings
  } deriving (Show, Generic, Binary)


-- | Prepare the sentence for further processing:
--
--   * Remove tokens with ID ranges
--   * Add dummy root token
--
-- NOTE: This function is dangaruous in the sense that it may discard MWE
-- annotations assigned to tokens with ID ranges.  This step should be perhaps
-- made more explicit.
--
cleanUp :: (KnownNat d) => Sent d -> Sent d
cleanUp sent0 =
  discardMerged sent0
    { cuptSent = dummyRootTok : cuptSent sent0
    , wordEmbs = dummyRootEmb : wordEmbs sent0
    }


-- | Remove tokens with ID ranges
discardMerged :: Sent d -> Sent d
discardMerged sent0 
  = uncurry Sent . unzip
  $ filter cond 
  $ zip (cuptSent sent0) (wordEmbs sent0)
  where
    cond (tok, _emb) =
      case Cupt.tokID tok of
        Cupt.TokID _ -> True
        _ -> False


-- | An artificial root token.
dummyRootTok :: Cupt.Token
dummyRootTok = Cupt.Token
  { tokID = Cupt.TokID 0
  , orth = ""
  , lemma = ""
  , upos = dummyRootPOS
  , xpos = ""
  , feats = M.empty
  , dephead = dummyRootParID
  , deprel = dummyRootDepRel
  , deps = ""
  , misc = ""
  , mwe = []
  }


-- | ID to refer to the parent of the artificial root node.
dummyRootParID :: Cupt.TokID
dummyRootParID = Cupt.TokID (-1)


-- | Embedding of the dummy root.
dummyRootEmb :: (KnownNat d) => R d
dummyRootEmb = 0


-- | Dummy root POS dummy label
dummyRootPOS :: T.Text
dummyRootPOS = "DUMMY-ROOT-POS"


-- | Dummy root DepRel dummy label
dummyRootDepRel :: T.Text
dummyRootDepRel = "DUMMY-ROOT-DEPREL"


----------------------------------------------
-- Sentence -> Graph conversion
----------------------------------------------


-- | Input information stored in a node
data Node d = Node
  { tok :: Cupt.Token
    -- ^ The corresponding token
  , emb :: R d
    -- ^ Word embedding
  }


-- | Trasform the sentence into the corresponding graph.
toGraph :: Sent d -> Graph.Graph (Node d) ()
toGraph sent =

  createElem nodes arcs

  where

    -- A map from token IDs to tokens
    tokMap = M.fromList
      [ (Cupt.tokID tok, tok)
      | tok <- cuptSent sent
      ]
    -- The parent of the given token
    tokPar tok = tokMap M.! Cupt.dephead tok
    -- Is the token a root?
    isRoot tok = not $ M.member (Cupt.dephead tok) tokMap

    -- Graph nodes: a list of token IDs and the corresponding `Node`s
    nodes = do
      (tok, vec) <- zipSafe
        (cuptSent sent)
        (wordEmbs sent)
      let node = Node tok vec
      return (tokID tok, node)

    -- Arcs of the graph
    arcs = do
      tok <- cuptSent sent
      -- Check the token is not a root
      guard . not $ isRoot tok
      let par = tokPar tok
      return (tokID tok, tokID par)


-- | Token ID
tokID :: Cupt.Token -> Int
tokID tok =
  case Cupt.tokID tok of
    Cupt.TokID i -> i
    Cupt.TokIDRange _ _ ->
      error "MWE.Sent.tokID: token ID ranges not supported"


-- | Create a dataset element based on nodes and arcs.
--
-- Works under the assumption that incoming/outgoing arcs are ,,naturally''
-- ordered in accordance with their corresponding vertex IDs (e.g., for two
-- children nodes x, v, the node with lower ID precedes the other node).
--
-- TODO: is this assumption still required?  We sort the arcs, so maybe not?
--
createElem
  :: [(G.Vertex, Node d)]
  -> [Graph.Arc]
  -> Graph.Graph (Node d) ()
createElem nodes arcs0 =
  graph
  where
    arcs = List.sort arcs0
    vertices = [v | (v, _) <- nodes]
    gStr = G.buildG
      (minimum vertices, maximum vertices)
      arcs
    graph = verify . Graph.mkAsc $ Graph.Graph
      { Graph.graphStr = gStr
      , Graph.graphInv = G.transposeG gStr
      , Graph.nodeLabelMap = M.fromList nodes
      , Graph.arcLabelMap = M.fromList $ map (,()) arcs
      }
    verify g
      | Graph.isAsc g = g
      | otherwise = error "MWE.createElem: constructed graph not ascending!"


----------------------------------------------
-- Utils
----------------------------------------------


-- | TODO: move to some utility module?
zipSafe :: [a] -> [b] -> [(a, b)]
zipSafe xs ys
  | length xs == length ys = zip xs ys
  | otherwise = error "zipSafe: length not the same!"
