{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}

{-# OPTIONS_GHC -fno-warn-orphans #-}


-- | The top-level module of the VMWE identification tool.  Provides plumbing
-- between various more specialized modules:
--
--   * `Net.Graph` -- ANN over graphs
--   * `Format.Cupt` -- .cupt format support
--   * `Numeric.SGD` -- stochastic gradient descent (external library)
--   * ...


module MWE 
  ( Sent(..)

  -- * New
  , Net.newO
  , Net.Typ

  -- * Training
  , Config(..)
  , Method(..)
  , trainO
  , depRelsIn
  , posTagsIn

  -- * Tagging
  , TagConfig(..)
  , tag
  , tagMany
  , tagManyO
  ) where


import           Prelude hiding (elem)

import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat)
-- import qualified GHC.TypeNats as Nats

import           Control.Monad (guard, forM_)
import           Control.DeepSeq (NFData)

-- import qualified System.Directory as D
-- import           System.FilePath ((</>))

import           Numeric.LinearAlgebra.Static.Backprop (R)
-- import qualified Numeric.LinearAlgebra.Static as LA
-- import qualified Numeric.LinearAlgebra as LAD
import qualified Numeric.Backprop as BP

import           Dhall -- (Interpret)
import           Dhall.Core (Expr(..))
import qualified Dhall.Map as Map
-- import qualified Data.Aeson as JSON

import qualified Data.Foldable as Fold
import           Data.Semigroup (Max(..))
import qualified Data.Graph as G
import qualified Data.Map.Strict as M
import qualified Data.Ord as Ord
-- import           Data.Function (on)
import qualified Data.List as List
import qualified Data.Set as S
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.Text.Lazy.IO as L
import           Data.Binary (Binary)
-- import qualified Data.Binary as Bin
-- import qualified Data.ByteString.Lazy as B
-- import           Codec.Compression.Zlib (compress, decompress)
-- import qualified Data.IORef as R

import qualified Format.Cupt as Cupt
import qualified Net.Graph as Net
import qualified Graph
import           Net.Graph (Elem)
import qualified Numeric.SGD as SGD
import qualified Numeric.SGD.Type as SGD
import qualified Numeric.SGD.ParamSet as SGD
import qualified Numeric.SGD.DataSet as SGD
import qualified Numeric.SGD.Momentum as Mom
import qualified Numeric.SGD.AdaDelta as Ada
import qualified Numeric.SGD.Adam as Adam
-- import qualified SGD as SGD

-- import qualified Net.Graph.BiComp as Bi
import qualified Net.Graph.QuadComp as Q

-- import Debug.Trace (trace)


----------------------------------------------
-- Data
----------------------------------------------


-- | Input sentence
data Sent d = Sent
  { cuptSent :: Cupt.Sent
    -- ^ The .cupt sentence
  , wordEmbs :: [R d]
    -- ^ The corresponding word embeddings
  } deriving (Show, Generic, Binary)



-- | Convert the given .cupt file to a training dataset element.  The token IDs
-- are used as vertex identifiers in the resulting graph.
--
--   * @d@ -- embedding size (dimension)
--
mkElem
  :: (KnownNat d)
  => (Cupt.MweTyp -> Bool)
    -- ^ MWE type (category) selection method
  -> Sent d
    -- ^ Input sentence
  -> Elem d DepRel POS
mkElem mweSel sent =

  createElem nodes arcs

  where

    -- Cupt sentence with ID range tokens removed
    cuptSentF = filter
      (\tok -> case Cupt.tokID tok of
                 Cupt.TokID _ -> True
                 _ -> False
      )
      (cuptSent sent)

    -- A map from token IDs to tokens
    tokMap = M.fromList
      [ (Cupt.tokID tok, tok)
      | tok <- cuptSentF 
      ]
    -- The parent of the given token
    tokPar tok = tokMap M.! Cupt.dephead tok
    -- Is the token the root?
    isRoot tok = Cupt.dephead tok == Cupt.TokID 0
    -- The IDs of the MWEs present in the given token
    getMweIDs tok
      = S.fromList
      . map fst
      . filter (mweSel . snd)
      $ Cupt.mwe tok

    -- Graph nodes: a list of token IDs and the corresponding vector embeddings
    nodes = do
      (tok, vec) <- zip cuptSentF (wordEmbs sent)
      let node = Graph.Node
            { nodeEmb = vec
            , nodeLab = Cupt.upos tok
            -- | TODO: could be lemma?
            , nodeLex = Cupt.orth tok
            }
      return (tokID tok, node)
    -- Labeled arcs of the graph
    arcs = do
      tok <- cuptSentF
      -- Check the token is not a root
      guard . not $ isRoot tok
      let par = tokPar tok
          -- The arc value is `True` if the token and its parent are annoted as
          -- a part of the same MWE of the appropriate MWE category
          arcVal =
            if (not . S.null)
                 ((getMweIDs tok) `S.intersection` (getMweIDs par))
               then True
               else False
      return ((tokID tok, tokID par), Cupt.deprel tok, arcVal)


-- | Create a dataset element based on nodes and labeled arcs.
--
-- Works under the assumption that incoming/outgoing arcs are ,,naturally''
-- ordered in accordance with their corresponding vertex IDs (e.g., for two
-- children nodes x, v, the node with lower ID preceds the other node).
--
createElem
  :: (KnownNat d)
  -- => [(G.Vertex, R d, POS)]
  => [(G.Vertex, Graph.Node d POS)]
  -> [(Graph.Arc, DepRel, Bool)]
  -> Elem d DepRel POS
createElem nodes arcs0 = Net.Elem
  { graph = graph
  , labMap = valMap
  }
  where
    arcs = List.sortBy (Ord.comparing _1) arcs0
    vertices = [v | (v, _) <- nodes]
    gStr = G.buildG
      (minimum vertices, maximum vertices)
      (map _1 arcs)
    graph = verify . Graph.mkAsc $ Graph.Graph
      { Graph.graphStr = gStr
      , Graph.graphInv = G.transposeG gStr
      , Graph.nodeLabelMap = M.fromList $ nodes
          -- map (\(x, e, pos) -> (x, Graph.Node e pos)) nodes
      , Graph.arcLabelMap = M.fromList $
          map (\(x, y, _) -> (x, y)) arcs
      }
    valMap = M.fromList $ do
      (arc, _arcLab, isMwe) <- arcs
      return 
        ( arc
        , if isMwe then 1.0 else 0.0
--         , if isMwe
--              then mwe
--              else notMwe
        )
    _1 (x, _, _) = x
    -- _2 (_, y, _) = y
    -- _3 (_, _, z) = z
    verify g
      | Graph.isAsc g = g
      | otherwise = error "MWE.createElem: constructed graph not ascending!"


-- -- | Is MWE or not?
-- mwe, notMwe :: R 2
-- notMwe = LA.vector [1, 0]
-- mwe = LA.vector [0, 1]


-- | Token ID
tokID :: Cupt.Token -> Int
tokID tok =
  case Cupt.tokID tok of
    Cupt.TokID i -> i
    Cupt.TokIDRange _ _ ->
      error "MWE.tokID: token ID ranges not supported"


----------------------------------------------
-- Training
----------------------------------------------


-- Some orphan `Interpret` instances
instance Interpret Mom.Config
instance Interpret Ada.Config
instance Interpret Adam.Config
instance Interpret SGD.Config
-- instance Interpret SGD.Method


data Method
  = Momentum Mom.Config
  | AdaDelta Ada.Config
  | Adam Adam.Config
  deriving (Generic)

instance Interpret Method where
  autoWith _ = rmUnion_1 genericAuto


data Config = Config
  { sgd :: SGD.Config
  , method :: Method
  } deriving (Generic)

instance Interpret Config


-- | Select the sgd method
toSGD
  :: (SGD.ParamSet p)
  => Method
    -- ^ SGD method
  -> (e -> p -> p)
    -- ^ Gradient
  -> SGD.SGD IO e p
toSGD method grad =
  case method of
    Momentum cfg -> Mom.momentum cfg grad
    AdaDelta cfg -> Ada.adaDelta cfg grad
    Adam cfg -> Adam.adam cfg grad


-- | Depdency relation
type DepRel = T.Text


-- | POS tag
type POS = T.Text


-- | Train the MWE identification network.
train
  :: ( KnownNat d
     , Q.QuadComp d DepRel POS comp
     , SGD.ParamSet comp, NFData comp
     )
  => Config
    -- ^ General training confiration
  -> Cupt.MweTyp
    -- ^ Selected MWE type (category)
  -> [Sent d]
    -- ^ Training dataset
  -> comp
--   -> Net.Param 300
    -- ^ Initial network
  -> IO comp
--   -> IO (Net.Param 300)
train Config{..} mweTyp cupt net0 = do
  -- dataSet <- mkDataSet (== mweTyp) tmpDir cupt
  let cupt' = map (mkElem (== mweTyp)) cupt
  net' <- SGD.withDisk cupt' $ \dataSet -> do
    putStrLn $ "# Training dataset size: " ++ show (SGD.size dataSet)
    -- net0 <- Net.new 300 2
    -- trainProgSGD sgd dataSet globalDepth net0
    SGD.runIO sgd
      (toSGD method $ SGD.batchGradPar gradient)
      quality dataSet net0
--   Net.printParam net'
  return net'
  where
    gradient x = BP.gradBP (Net.netErrorQ [x])
    quality x = BP.evalBP (Net.netErrorQ [x])


-- | Extract dependeny relations present in the given dataset.
depRelsIn :: [Cupt.GenSent mwe] -> S.Set DepRel
depRelsIn =
  S.fromList . concatMap extract 
  where
    extract = map Cupt.deprel


-- | Extract dependeny relations present in the given dataset.
posTagsIn :: [Cupt.GenSent mwe] -> S.Set POS
posTagsIn =
  S.fromList . concatMap extract 
  where
    extract = map Cupt.upos


----------------------------------------------
-- Working with opaque network architecture
----------------------------------------------


-- | Train the opaque MWE identification network.
trainO
  :: Config
    -- ^ General training confiration
  -> Cupt.MweTyp
    -- ^ Selected MWE type (category)
  -> [Sent 300]
    -- ^ Training dataset
  -> Net.Opaque 300 DepRel POS
    -- ^ Initial network
  -> IO (Net.Opaque 300 DepRel POS)
trainO cfg mweTyp cupt net =
  case net of
    Net.Opaque t p -> Net.Opaque t <$> train cfg mweTyp cupt p


-- | Tag sentences with the opaque network.
tagManyO
  :: TagConfig
  -> Net.Opaque 300 DepRel POS
  -> [Sent 300]
  -> IO ()
tagManyO cfg = \case
  Net.Opaque _ p -> tagMany cfg p


----------------------------------------------
-- Tagging
----------------------------------------------


-- | Tagging configuration
data TagConfig = TagConfig
  { mweThreshold :: Double
    -- ^ The minimum probability to consider an arc a MWE component
    -- (with 0.5 being the default)
  , mweTyp :: Cupt.MweTyp
    -- ^ MWE category to annotate
  } deriving (Show, Eq, Ord)


-- | Tag sentences based on the given configuration and multi/quad-affine
-- network.  The output is directed to stdout.
tagMany
  :: ( KnownNat d
     , Q.QuadComp d DepRel POS comp
     -- , SGD.ParamSet comp, NFData comp
     )
  => TagConfig
  -> comp
                      -- ^ Network parameters
  -> [Sent d]       -- ^ Cupt sentences
  -> IO ()
tagMany tagCfg net cupt = do
  forM_ cupt $ \sent -> do
    T.putStr "# "
    T.putStrLn . T.unwords $ map Cupt.orth (cuptSent sent)
    tag tagCfg net sent


-- | Tag a single sentence with the given network.
tag
  :: ( KnownNat d
     , Q.QuadComp d DepRel POS comp
     -- , SGD.ParamSet comp, NFData comp
     )
  => TagConfig
  -- -> Net.Param 300 DepRel POS
  -> comp
                      -- ^ Network parameters
  -> Sent d           -- ^ Cupt sentence
  -> IO ()
tag tagCfg net sent = do
  L.putStrLn $ Cupt.renderPar [Cupt.abstract sent']
  where
    elem = mkElem (const False) sent
    arcMap = Net.evalQ net (Net.graph elem)
    sent' = annotate tagCfg (cuptSent sent) arcMap


----------------------------------------------
-- Annotation
----------------------------------------------


-- | Annotate the sentence with the given MWE type, given the network
-- evaluation results.
annotate
  :: TagConfig
  -> Cupt.Sent
    -- ^ Input .cupt sentence
  -- -> M.Map Graph.Arc (R 2)
  -> M.Map Graph.Arc Double
    -- ^ Net evaluation results
  -> Cupt.Sent
annotate TagConfig{..} cupt arcMap =

  map enrich cupt

  where

    -- Enrich the token with new MWE information
    enrich tok = Prelude.maybe tok id $ do
      Cupt.TokID i <- return (Cupt.tokID tok)
      mweId <- M.lookup i mweIdMap
      let newMwe = (mweId, mweTyp)
      return tok {Cupt.mwe = newMwe : Cupt.mwe tok}

    -- Determine the set of MWE arcs
    arcSet = S.fromList $ do
      (arc, v) <- M.toList arcMap
      guard $ isMWE v
      return arc

--     isMWE = (>= 0.5)
    isMWE = (>= mweThreshold)
--     isMWE statVect =
--       let vect = LA.unwrap statVect
--           val = vect `LAD.atIndex` 1
--        in val > 0.5

    -- Determine the mapping from nodes to new MWE id's
    ccs = findConnectedComponents arcSet
    mweIdMap = M.fromList . concat $ do
      (cc, mweId) <- zip ccs [maxMweID cupt + 1 ..]
      (v, w) <- S.toList cc
      return [(v, mweId), (w, mweId)]


-- | Given a set of graph arcs, determine all the connected arc subsets in the
-- corresponding graph.
findConnectedComponents
  :: S.Set Graph.Arc
  -> [S.Set Graph.Arc]
findConnectedComponents arcSet
  | S.null arcSet = []
  | otherwise
      -- Some components can be empty!  Perhaps because the graph is sparse?
      = filter (not . S.null)
      $ map (arcsInTree arcSet) (G.components graph)
  where
    vertices = S.toList (nodesIn arcSet)
    graph = G.buildG
      (minimum vertices, maximum vertices)
      (S.toList arcSet)


-- | Determine the set of arcs in the given connected graph component.
-- TODO: This could be done much more efficiently!
arcsInTree
  :: S.Set Graph.Arc
    -- ^ The set of all arcs
  -> G.Tree G.Vertex
    -- ^ Connected component
  -> S.Set Graph.Arc
arcsInTree arcSet cc = S.fromList $ do
  (v, w) <- S.toList arcSet
  guard $ v `S.member` vset
  guard $ w `S.member` vset
  return (v, w)
  where
    vset = S.fromList (Fold.toList cc)


-- | The set of nodes in the given arc set.
nodesIn :: S.Set Graph.Arc -> S.Set G.Vertex
nodesIn arcSet =
  (S.fromList . concat)
    [[v, w] | (v, w) <- S.toList arcSet]


-- | Determine the maximum mwe ID present in the given sentence.
maxMweID :: Cupt.Sent -> Int
maxMweID =
  getMax . Fold.foldMap mweID
  where
    mweID tok = case Cupt.mwe tok of
      [] -> Max 0
      xs -> Max . maximum $ map fst xs


----------------------------------------------
-- Dhall Utils
----------------------------------------------


-- | Remove the top-level _1 fields from the given union type.
rmUnion_1 :: Type a -> Type a
rmUnion_1 typ = typ
  { extract = \expr -> extract typ (add1 expr)
  , expected = rm1 (expected typ)
  }
  where
    -- Add _1 to union expression
    add1 expr =
      case expr of
        Union m -> Union (fmap addField_1 m)
        UnionLit key val m -> UnionLit key (addField_1 val) (fmap addField_1 m)
        _ -> expr
    -- Remove _1 from union epxression
    rm1 expr =
      case expr of
        Union m -> Union (fmap rmField_1 m)
        UnionLit key val m -> UnionLit key (rmField_1 val) (fmap rmField_1 m)
        _ -> expr


-- | Add _1 in the given record expression.
addField_1 :: Expr s a -> Expr s a
addField_1 expr =
  case expr of
    RecordLit m -> RecordLit (Map.singleton "_1" (RecordLit m))
    Record m -> Record (Map.singleton "_1" (Record m))
    _ -> expr


-- | Remove _1 from the given record expression.
rmField_1 :: Expr s a -> Expr s a
rmField_1 expr =
  case expr of
    RecordLit m -> Prelude.maybe (RecordLit m) id $ do
      guard $ Map.keys m == ["_1"]
      RecordLit m' <- Map.lookup "_1" m
      return (RecordLit m')
    Record m -> Prelude.maybe (Record m) id $ do
      guard $ Map.keys m == ["_1"]
      Record m' <- Map.lookup "_1" m
      return (Record m')
    _ -> expr
